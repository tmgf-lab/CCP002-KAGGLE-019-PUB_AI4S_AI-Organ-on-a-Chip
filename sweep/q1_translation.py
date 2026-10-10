"""Q1 tree search: data -> processing -> model -> exploration -> pixel_r (see sweep/configs/q1_translation.yaml).

Each path trains in phases (25% -> 50% -> 100% of the training fields) and is scored on the validation
fields after each phase. Optuna's TPE sampler chooses the next path from the results so far; the
successive-halving pruner stops paths that are in the worse half at a checkpoint.

    python -m sweep.q1_translation search   --gpu 0 [--smoke]     # run in parallel: one process per GPU
    python -m sweep.q1_translation baseline --gpu 0 --seeds 1 2 3  # noise of the current model
    python -m sweep.q1_translation final    --gpu 0 --top 3 --seeds 1 2 3
    python -m sweep.q1_translation report                          # tables from the study + final runs
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from data import LazyCropDataset, input_spec, load_plate, norm_params, normalise, predict  # noqa: E402
from model import build_model  # noqa: E402

# VCP_Q1_CONFIG: another tree file, e.g. configs/q1_demo.yaml (small tree that fits a laptop GPU, for the video demo)
CFG = yaml.safe_load((Path(__file__).parent / "configs" / os.environ.get("VCP_Q1_CONFIG", "q1_translation.yaml")).read_text())
OUT = C.RUNS / "q1_translation"
BASELINE = {"data": "brightfield_a549", "processing": "plate_scaling", "model": "unet",
            "width": 32, "depth": 4, "attention": False, "upsampling": "transpose"}
DATA_PLATES = {"brightfield_a549": ["BR00117015"], "brightfield_a549+u2os": ["BR00117015", "BR00117024"],
               "brightfield_a549_2plates": ["BR00117015", "BR00117016"]}   # 2plates: final-stage variant only


def path_string(p):
    expl = "encoder" if p["model"] != "unet" else f"width {p['width']}, depth {p['depth']}"
    expl += f", attention {'on' if p['attention'] else 'off'}" + (f", {p['upsampling']}" if p["model"] == "unet" else "")
    return f"{p['data']} -> {p['processing']} -> {p['model']} -> {expl} -> pixel_r"


def run_config(p):
    """Search path -> training config (also what load_model() reads back from train_log.json)."""
    return {"inputs": None, "bf_norm": "image" if p["processing"].startswith("image") else "plate",
            "aug": "augmentation" in p["processing"], "target_transform": "log" if "log" in p["processing"] else "linear",
            "model": p["model"], "width": p.get("width", 32), "depth": p.get("depth", 4), "base": p.get("width", 32),
            "attention": p["attention"], "upsampling": p.get("upsampling", "transpose"),
            "train_plates": DATA_PLATES[p["data"]], "qc": p.get("cleaning") == "focus_qc"}


def suggest(trial):
    t = CFG["tree"]
    p = {"data": trial.suggest_categorical("data", t["data"]),
         "processing": trial.suggest_categorical("processing", t.get("processing", t.get("transform"))),  # node named "transform" in the yaml
         "model": trial.suggest_categorical("model", t["model"])}
    if p["model"] == "unet":
        p["width"] = trial.suggest_categorical("width", t["exploration"]["width"])
        p["depth"] = trial.suggest_categorical("depth", t["exploration"]["depth"])
        p["upsampling"] = trial.suggest_categorical("upsampling", t["exploration"]["upsampling"])
    p["attention"] = trial.suggest_categorical("attention", [a == "on" for a in t["exploration"]["attention"]])
    return p


# ------------------------------------------------------------------------------------------ training
class Validator:
    """Mean per-channel Pearson r on whole fields; cached normalised fields."""

    def __init__(self, split, n_max, seed=0):
        arr, idx = load_plate(C.TRAIN_PLATES[0])
        lo, hi = norm_params(C.TRAIN_PLATES[0])
        sel = np.where(idx.split == split)[0] if split == "val" else np.where(idx.split.isin(["train", "control"]))[0]
        if len(sel) > n_max:
            sel = np.sort(np.random.default_rng(seed).choice(sel, n_max, replace=False))
        self.fields = [normalise(np.asarray(arr[i]), lo, hi) for i in sel]

    def __call__(self, model, cfg, dev):
        model.eval()
        rs = []
        for x in self.fields:
            v = predict(model, cfg, x, dev)
            rs.append([np.corrcoef(v[c].ravel(), x[c].ravel())[0, 1] for c in range(len(C.TARGET_CH))])
        model.train()
        return float(np.nanmean(rs))


def focus_qc(ds, log=print):
    """Drop training fields with the weakest focus (bottom 5% variance of Laplacian of the mid bright-field plane,
    per plate) or with saturated bright-field (> 1% of pixels at the plate's 99.9th percentile ceiling)."""
    from scipy.ndimage import laplace
    bf = C.INPUT_CH[1]
    rows = []
    for j, (arr, i, lo, hi) in enumerate(ds.entries):
        img = np.asarray(arr[i, bf, ::2, ::2], np.float32)
        rows.append((j, id(arr), float(laplace(img).var()), float((img >= hi[bf, 0, 0]).mean())))
    df = pd.DataFrame(rows, columns=["j", "plate", "focus", "sat"])
    df["low"] = df.groupby("plate").focus.transform(lambda f: f < f.quantile(0.05))
    keep = df[~df.low & (df.sat <= 0.01)].j.to_numpy()
    log(f"    focus QC: kept {len(keep)} of {len(df)} training fields")
    ds.entries = [ds.entries[j] for j in keep]
    return ds


def make_dataset(cfg, inputs):
    ds = LazyCropDataset(cfg["train_plates"], ["train", "control"], inputs=inputs, bf_norm=cfg["bf_norm"],
                         aug=cfg["aug"], target_transform=cfg["target_transform"])
    return focus_qc(ds) if cfg.get("qc") else ds


def train(cfg, phases, seed, dev, val, train_val, trial=None, out_dir=None, bs=12, lr=2e-3, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    inputs, _ = input_spec(cfg)
    ds = make_dataset(cfg, inputs)
    n_all = ds._n_fields()
    order = np.random.default_rng(seed).permutation(n_all)          # nested subsets: 25% c 50% c 100%
    model = build_model(cfg, len(inputs), len(C.TARGET_CH)).to(dev)
    total = sum(ph["steps"] for ph in phases)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total, pct_start=0.05)
    cuda = str(dev).startswith("cuda")  # CPU works too (float32; for demos and smoke tests)
    bf16 = cuda and torch.cuda.get_device_capability()[0] >= 8
    scaler = torch.amp.GradScaler("cuda", enabled=cuda and not bf16)
    hist, t0, step, bad = [], time.time(), 0, 0
    for k, ph in enumerate(phases):
        ds.idx = np.sort(order[:max(1, int(round(ph["fraction"] * n_all)))])  # indices only: no copy
        ds.n = ph["steps"] * bs
        dl = torch.utils.data.DataLoader(ds, batch_size=bs, num_workers=int(os.environ.get("VCP_WORKERS", 3)), pin_memory=True, drop_last=True,
                                         worker_init_fn=lambda w: np.random.seed((seed * 1000 + k * 100 + w) % 2**32))
        for x, y in dl:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda" if cuda else "cpu", dtype=torch.bfloat16 if bf16 else torch.float16, enabled=cuda):
                pred = model(x)
            loss = F.l1_loss(pred.float(), y)
            if not torch.isfinite(loss):  # fp16 overflow on T4: skip this batch instead of crashing the run
                bad += 1
                opt.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                if bad > 50:
                    raise optuna.TrialPruned("non-finite loss") if trial else FloatingPointError("non-finite loss x50")
                continue
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
        v, tv = val(model, cfg, dev), train_val(model, cfg, dev)
        hist.append({"phase": k + 1, "fraction": ph["fraction"], "step": step, "val_r": v, "train_r": tv,
                     "gap": tv - v, "loss": loss.item(), "minutes": (time.time() - t0) / 60})
        log(f"    phase {k + 1}: {int(ph['fraction'] * 100)}% data, step {step}: val r {v:.4f} train r {tv:.4f} "
            f"({hist[-1]['minutes']:.1f} min)")
        if trial is not None:
            trial.report(v, k + 1)
            for key, val_ in hist[-1].items():
                trial.set_user_attr(f"p{k + 1}_{key}", val_)
            if k + 1 < len(phases) and trial.should_prune():
                log(f"    -> stopped at {int(ph['fraction'] * 100)}% data (worse half at this checkpoint)")
                raise optuna.TrialPruned()
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "best.pt")
        (out_dir / "train_log.json").write_text(json.dumps({"args": {**cfg, "seed": seed, "phases": phases},
                                                            "log": hist, "best_mean_r": hist[-1]["val_r"]}, indent=1))
    return hist


def phases_from_config(smoke):
    ph = [{"fraction": float(c["data"].rstrip("%")) / 100, "steps": c["steps"]} for c in CFG["search"]["checkpoints"]]
    if smoke:
        ph = [{**p, "steps": max(10, p["steps"] // 200)} for p in ph]
    return ph


def storage():
    OUT.mkdir(parents=True, exist_ok=True)
    return optuna.storages.RDBStorage(f"sqlite:///{OUT / 'study.db'}", engine_kwargs={"connect_args": {"timeout": 60}})


def make_sampler(seed=0):
    return optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True, constant_liar=True, n_startup_trials=8)


def make_study():
    return optuna.create_study(
        study_name="q1", storage=storage(), direction="maximize", load_if_exists=True,
        sampler=make_sampler(0),
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2, min_early_stopping_rate=0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["init", "search", "baseline", "final", "variants", "choose", "report", "probe",
                                     "replicate"])
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="tiny steps and few validation fields: checks the code only")
    ap.add_argument("--trials", type=int, default=None, help="trials for THIS process (default: config / 2)")
    ap.add_argument("--hours", type=float, default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--winner", help="replicate: q1_winner.json of the A549 study")
    args = ap.parse_args()
    if args.mode == "report":
        return report()
    if args.mode == "init":  # create the study once, before parallel workers start (avoids a schema race)
        study = make_study()
        if not study.trials:
            study.enqueue_trial(dict(BASELINE))
        print("study ready:", len(study.trials), "trial(s) queued")
        return

    dev = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    if dev != "cpu":
        torch.cuda.set_device(args.gpu)
    phases = phases_from_config(args.smoke)
    val = Validator("val", 8 if args.smoke else 96)
    train_val = Validator("train", 4 if args.smoke else 32)
    logf = open(OUT.mkdir(parents=True, exist_ok=True) or OUT / f"log_{args.mode}_gpu{args.gpu}.txt", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    if args.mode == "probe":  # minutes per 1,000 steps for the cheapest and the most expensive branches
        res = []
        for name, p in [("baseline", BASELINE), ("width64_depth5", {**BASELINE, "width": 64, "depth": 5}),
                        ("resnet34", {**BASELINE, "model": "unet_resnet34_pretrained", "processing": "image_scaling+augmentation"})]:
            h = train(run_config(p), [{"fraction": 1.0, "steps": 200}], 0, dev, val, train_val, log=log)
            res.append({"branch": name, "gpu": args.gpu, "min_per_1000_steps": h[-1]["minutes"] * 5,
                        "peak_gb": torch.cuda.max_memory_allocated() / 1e9})
            log(str(res[-1]))
        pd.DataFrame(res).to_csv(OUT / f"probe_gpu{args.gpu}.csv", index=False)
    elif args.mode == "search":
        # each worker gets its own sampler seed, so the two GPUs do not propose the same paths
        study = optuna.load_study(study_name="q1", storage=storage(), sampler=make_sampler(100 + args.gpu),
                                  pruner=make_study().pruner)

        def objective(trial):
            p = suggest(trial)
            trial.set_user_attr("path", path_string(p))
            seen = {tuple(sorted(t.params.items())) for t in trial.study.trials if t.number != trial.number
                    and not t.user_attrs.get("skipped") and t.state.name in ("COMPLETE", "PRUNED", "RUNNING")}
            if tuple(sorted(trial.params.items())) in seen:  # training is deterministic: a repeat adds nothing
                trial.set_user_attr("skipped", "duplicate")
                log(f"trial {trial.number}: repeat of an evaluated path, skipped")
                raise optuna.TrialPruned()
            log(f"trial {trial.number}: {path_string(p)}")
            hist = train(run_config(p), phases, 0, dev, val, train_val, trial=trial,
                         out_dir=OUT / "trials" / f"t{trial.number:03d}", log=log)
            log(f"    -> completed: validation r {hist[-1]['val_r']:.4f}")
            return hist[-1]["val_r"]

        optuna.logging.set_verbosity(optuna.logging.WARNING)  # the log lines above say the same, more briefly
        n = args.trials or CFG["search"]["paths_to_try"] // CFG["search"]["parallel"]
        hours = args.hours or CFG["search"]["time_limit_hours"]
        study.optimize(objective, n_trials=n, timeout=hours * 3600, catch=(torch.OutOfMemoryError,))
    elif args.mode == "baseline":
        for s in args.seeds:
            log(f"baseline seed {s}: {path_string(BASELINE)}")
            train(run_config(BASELINE), phases, s, dev, val, train_val, out_dir=OUT / "baseline" / f"seed{s}", log=log)
    elif args.mode == "final":
        study = optuna.load_study(study_name="q1", storage=storage())
        done = sorted([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE], key=lambda t: -t.value)
        distinct, seen = [], set()
        for t in done:  # the search can propose the same path more than once: keep the best trial of each path
            key = tuple(sorted(t.params.items()))
            if key not in seen and t.params != {k: v for k, v in BASELINE.items() if k in t.params}:
                seen.add(key)
                distinct.append(t)
        for t in distinct[:args.top]:
            p = {**BASELINE, **t.params}
            for s in args.seeds:
                d = OUT / "final" / f"t{t.number:03d}_seed{s}"
                if (d / "train_log.json").exists():
                    continue
                log(f"final t{t.number} seed {s}: {path_string(p)}")
                safe_train(run_config(p), phases, s, dev, val, train_val, d, log)
        report()
    elif args.mode == "variants":  # best final path x {1, 2 A549 plates} x {no QC, focus QC}
        best = best_final()
        if best is None:
            log("variants: no completed final runs")
            return
        for data in ("brightfield_a549", "brightfield_a549_2plates"):
            for cleaning in ("none", "focus_qc"):
                if data == "brightfield_a549" and cleaning == "none":
                    continue  # = the final runs of the best path
                p = {**best["path"], "data": data, "cleaning": cleaning}
                for s in args.seeds:
                    d = OUT / "variants" / f"{best['config']}_{data}_{cleaning}_seed{s}"
                    if (d / "train_log.json").exists():
                        continue
                    log(f"variant {data}/{cleaning} seed {s}: {path_string(p)}")
                    safe_train(run_config(p), phases, s, dev, val, train_val, d, log)
        report()
    elif args.mode == "choose":
        choose(dev, val, log)
    elif args.mode == "replicate":
        replicate(args.winner, phases, dev, val, train_val, log, args.seeds)


def safe_train(cfg, phases, seed, dev, val, train_val, out_dir, log):
    """One run failing (e.g. numerical overflow) must not stop the remaining runs of the stage."""
    try:
        return train(cfg, phases, seed, dev, val, train_val, out_dir=out_dir, log=log)
    except Exception as e:  # noqa: BLE001
        log(f"    RUN FAILED ({type(e).__name__}: {e}); continuing with the next run")
        torch.cuda.empty_cache()


def best_final():
    """Best final-stage configuration by mean validation r over its seeds."""
    runs = sorted((OUT / "final").glob("*/train_log.json")) if (OUT / "final").exists() else []
    if not runs:
        return None
    df = pd.DataFrame([{"config": f.parent.name.rsplit("_seed", 1)[0], "val_r": json.loads(f.read_text())["log"][-1]["val_r"]}
                       for f in runs])
    cfgname = df.groupby("config").val_r.mean().idxmax()
    study = optuna.load_study(study_name="q1", storage=storage())
    t = study.trials[int(cfgname[1:])]
    return {"config": cfgname, "path": {**BASELINE, **t.params}, "val_r": float(df.groupby("config").val_r.mean().max())}


def choose(dev, val, log=print):
    """Winner = best mean validation r over all final / variant / baseline configurations (>= 2 seeds); then the
    inference mode (single model, test-time augmentation, seed ensemble, both) also chosen on validation fields."""
    from data import load_model, predict_multi
    rows = []
    for group in ("final", "variants", "baseline"):
        for f in sorted((OUT / group).glob("*/train_log.json")) if (OUT / group).exists() else []:
            rows.append({"group": group, "dir": str(f.parent), "config": group + "/" + f.parent.name.rsplit("seed", 1)[0].rstrip("_"),
                         "val_r": json.loads(f.read_text())["log"][-1]["val_r"]})
    df = pd.DataFrame(rows)
    g = df.groupby("config").agg(val_r=("val_r", "mean"), sd=("val_r", "std"), n=("val_r", "size")).reset_index()
    g = g[g.n >= 2].sort_values("val_r", ascending=False)
    win = g.iloc[0].config
    dirs = df[df.config == win].sort_values("val_r", ascending=False).dir.tolist()
    models = [load_model(Path(d), dev=dev) for d in dirs]
    cfg = models[0][1]
    res = {}
    for name, ms, tta in [("single_best_seed", [models[0][0]], False), ("tta", [models[0][0]], True),
                          ("seed_ensemble", [m for m, _ in models], False), ("seed_ensemble+tta", [m for m, _ in models], True)]:
        rs = []
        for x in val.fields:
            v = predict_multi(ms, cfg, x, dev, tta)
            rs.append(np.nanmean([np.corrcoef(v[c].ravel(), x[c].ravel())[0, 1] for c in range(len(C.TARGET_CH))]))
        res[name] = float(np.mean(rs))
        log(f"  inference {name}: val r {res[name]:.4f}")
    inf = max(res, key=res.get)
    out = {"config": win, "run_dirs": dirs, "inference": inf, "inference_val_r": res,
           "ranking": g.head(10).to_dict("records")}
    (OUT / "q1_winner.json").write_text(json.dumps(out, indent=1))
    log(f"winner {win} ({len(dirs)} seeds), inference {inf}")


def replicate(winner_json, phases, dev, val, train_val, log=print, seeds=None):
    """Second-dataset study: the A549 winning path (same settings, same seeds, same inference mode; nothing re-chosen)
    retrained on the plates in config.TRAIN_PLATES (e.g. U2OS BR00117011). Writes <RUNS>/q1_replicate/q1_winner.json."""
    w = json.loads(Path(winner_json).read_text())
    out = C.RUNS / "q1_replicate"
    dirs = []
    for d in w["run_dirs"]:
        src = json.loads((Path(d) / "train_log.json").read_text())["args"]
        seed = src.pop("seed")
        src.pop("phases", None)
        cfg = {**src, "train_plates": list(C.TRAIN_PLATES)}  # the A549 plate(s) replaced by this dataset's plate(s)
        d_out = out / f"seed{seed}"
        if seeds is not None and seed not in seeds:  # another GPU process trains this seed
            continue
        if not (d_out / "train_log.json").exists():
            log(f"replicate seed {seed} on {cfg['train_plates']}: {json.dumps({k: cfg.get(k) for k in ('model', 'width', 'depth', 'upsampling', 'attention', 'qc')})}")
            safe_train(cfg, phases, seed, dev, val, train_val, d_out, log)
        if (d_out / "train_log.json").exists():
            dirs.append(str(d_out))
    if not dirs:
        raise RuntimeError("replicate: no seed trained successfully")
    out.mkdir(parents=True, exist_ok=True)
    rep = {"source_winner": w["config"], "train_plates": list(C.TRAIN_PLATES), "run_dirs": dirs, "inference": w["inference"],
           "val_r": [json.loads((Path(d) / "train_log.json").read_text())["log"][-1]["val_r"] for d in dirs]}
    (out / "q1_winner.json").write_text(json.dumps(rep, indent=1))
    log(f"replicate done: {len(dirs)} seeds, val r {rep['val_r']}")


def report():
    rows = []
    if (OUT / "study.db").exists():
        study = optuna.load_study(study_name="q1", storage=storage())
        for t in study.trials:
            r = {"trial": t.number, "state": t.state.name, "path": t.user_attrs.get("path", ""), "value": t.value}
            for k in (1, 2, 3):
                r[f"val_r_{k}"] = t.user_attrs.get(f"p{k}_val_r")
                r[f"gap_{k}"] = t.user_attrs.get(f"p{k}_gap")
                r[f"min_{k}"] = t.user_attrs.get(f"p{k}_minutes")
            rows.append(r)
        pd.DataFrame(rows).to_csv(OUT / "search_trials.csv", index=False)
    fin = []
    for group in ("baseline", "final", "variants"):
        for f in sorted((OUT / group).glob("*/train_log.json")) if (OUT / group).exists() else []:
            d = json.loads(f.read_text())
            fin.append({"group": group, "run": f.parent.name, "val_r": d["log"][-1]["val_r"],
                        "train_r": d["log"][-1]["train_r"], "gap": d["log"][-1]["gap"], "minutes": d["log"][-1]["minutes"]})
    if fin:
        fin = pd.DataFrame(fin)
        fin["config"] = fin.run.str.replace(r"_?seed\d+$", "", regex=True).replace("", "baseline")
        summ = fin.groupby(["group", "config"]).val_r.agg(["mean", "std", "count"]).reset_index().sort_values("mean", ascending=False)
        summ.to_csv(OUT / "final_summary.csv", index=False)
        print(summ.to_string(index=False))
    if rows:
        print(pd.DataFrame(rows)[["trial", "state", "value", "path"]].to_string(index=False))


if __name__ == "__main__":
    main()
