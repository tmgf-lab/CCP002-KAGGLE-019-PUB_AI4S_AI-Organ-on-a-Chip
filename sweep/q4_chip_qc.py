"""Q4 - chip quality control: is this organ-on-a-chip bright-field image 'good' or 'bad'?
Config: sweep/configs/q4_chip_qc.yaml. Search engine: sweep/engine.py. Statistics: sweep/stats.py.

Commands (run from the solution root):
    python -m sweep.q4_chip_qc embed    [--zip Z --xlsx X | --img-dir D] --emb-dir E [--reps ...] [--bench 20] [--smoke]
    python -m sweep.q4_chip_qc search   --emb-dir E --out O [--budget 1000 --workers 4 --hours 10] [--smoke]
    python -m sweep.q4_chip_qc finetune --emb-dir E --out O [--budget 24 --workers 2 --epochs 8] [--smoke]
    python -m sweep.q4_chip_qc assess   --emb-dir E --out O [--bayes auto|laplace|pymc] [--smoke]

embed    (GPU)  frozen features for every (representation, preprocessing) pair + 224x299 pixels; resumable.
search   (CPU)  one engine.Search over scope x preprocessing x representation x fusion x reduction x model
                (non-Bayesian, non-fine-tuned models). Score = out-of-fold ROC AUC, 5-fold CV grouped by experiment
                (StratifiedGroupKFold, same folds for every path) on nested subsets of experiments 25/50/100%.
                scope: only all_cell_types is searched; per_cell_type is evaluated for the best path in `assess`.
finetune (GPU)  engine.Search over preprocessing x trained_layers x lr for a fine-tuned ResNet18 (budget 24),
                same grouped CV and checkpoints; the best configuration is then refit on all data and saved.
assess   (CPU)  best paths: cluster-bootstrap AUC CI (resampling experiments), calibration, Brier, per-cell-type AUC,
                per_cell_type scope, leave-one-cell-type-out, leave-one-day-out, DeLong vs baseline, Bayesian logistic
                on the best 4 pipelines' features, Grad-CAM (fine-tuned) or DINOv2 patch-similarity maps, figures.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sweep import engine, stats  # noqa: E402
from sweep import q4_parts as P  # noqa: E402

LOCAL_OOC = ROOT.parent.parent / "data" / "external" / "ooc"
OUT = ROOT / "runs" / "q4_chip_qc"
VIRT_CSV = ROOT / "legacy" / "runs" / "chip1" / "chip_s2" / "chip_features.csv"  # single-plane model, legacy
MODELS = ["logistic_l1", "logistic_l2", "logistic_elasticnet", "svm", "random_forest", "hist_gradient_boosting",
          "mlp", "knn"]   # bayes_logistic -> assess; finetuned_cnn -> finetune command
BASELINE = {"scope": "all_cell_types", "preprocessing": "centre_crop", "representation": "handcrafted",
            "fusion": "single", "reduction": "none", "model": "hist_gradient_boosting"}
FT_BASELINE = {"preprocessing": "centre_crop", "trained_layers": "last_block", "lr": 3e-4}
SMOKE_REPS = ["handcrafted", "glcm_lbp", "resnet50"]
GREY, ORANGE, BG, INK = "#a9a8a2", "#eb6834", "#fcfcfb", "#3d3d3a"
warnings.filterwarnings("ignore")


def is_kaggle():
    return Path("/kaggle").exists()


def find_input(name):
    """First file called `name` under /kaggle/input (Kaggle) or None."""
    if is_kaggle():
        for p in Path("/kaggle/input").rglob(name):
            return str(p)
    return None


# ================================================================================================ embed
def cmd_embed(a):
    emb = Path(a.emb_dir)
    emb.mkdir(parents=True, exist_ok=True)
    zip_path = a.zip or find_input("OOC_image_dataset.zip")
    xlsx = a.xlsx or find_input("OOC_datasheet.xlsx") or str(LOCAL_OOC / "OOC_datasheet.xlsx")
    if not zip_path and not a.img_dir:
        a.img_dir = str(LOCAL_OOC / "OOC_image_dataset")
    t0 = time.time()
    idx = P.build_index(xlsx, zip_path=zip_path, img_dir=None if zip_path else a.img_dir)
    print(f"index: {len(idx)} images, {idx.experiment.nunique()} experiments ({time.time() - t0:.0f}s)", flush=True)
    if a.smoke:
        idx = idx.iloc[engine.nested_subset(idx.experiment.to_numpy(), 0.10, seed=_label_seed(idx, (0.10,)))]
        idx = idx.reset_index(drop=True)
    reps = a.reps or (SMOKE_REPS if a.smoke else [r for r in P.ALL_REPS if r != "virtual_stain_features"])
    if a.bench:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        res = P.benchmark(idx.src.tolist(), zip_path, reps, a.bench, a.workers)
        res["extrapolation"] = _extrapolate(res, n=3072)
        res["device"], res["workers"] = dev, a.workers
        (emb / "bench.json").write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return
    old = emb / "index.csv"
    if old.exists() and pd.read_csv(old, dtype={"imageID": str}).imageID.tolist() != idx.imageID.tolist():
        sys.exit(f"{emb} holds embeddings of a different image set; use another --emb-dir")
    idx.drop(columns=["src"]).to_csv(old, index=False)
    has_virt = P.virtual_features(emb, idx, a.virt_csv or find_input("chip_features.csv") or str(VIRT_CSV))
    print(f"virtual_stain_features: {'ok' if has_virt else 'not available'}", flush=True)
    import torch
    n_gpu = torch.cuda.device_count()
    srcs = idx.src.tolist()
    if n_gpu > 1:  # one process per GPU, chunks split between them
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        procs = [ctx.Process(target=P.embed_shard, args=(str(emb), srcs, zip_path, reps, P.PREPS, f"cuda:{i}",
                                                        max(1, a.workers // n_gpu), a.chunk, i, n_gpu))
                 for i in range(n_gpu)]
        [p.start() for p in procs]
        [p.join() for p in procs]
    else:
        P.embed_shard(str(emb), srcs, zip_path, reps, P.PREPS, "cuda:0" if n_gpu else "cpu", a.workers, a.chunk)
    P.merge(emb, reps, P.PREPS, len(idx))
    done = sorted(f.name for f in emb.glob("*.npy"))
    meta = {"n_images": len(idx), "smoke": a.smoke, "reps": reps, "preps": P.PREPS,
            "models": {r: P.DEEP[r][:2] for r in reps if r in P.DEEP},
            "input_hw": {r: P.DEEP[r][2] for r in reps if r in P.DEEP}, "files": done,
            "wall_s": round(time.time() - t0, 1), "n_gpu": n_gpu}
    (emb / "q4_meta.json").write_text(json.dumps(meta, indent=1))
    print(f"embed done in {time.time() - t0:.0f}s: {len(done)} arrays", flush=True)


def _extrapolate(res, n=3072, t4_tflops=10.0, n_cores=4):
    """T4: FLOPs x 7 forwards / effective fp16 throughput (10 TFLOPS assumed: ~15% of the 65 TFLOPS peak)."""
    fwd = 7 * n
    deep = {r: v for r, v in res.items() if isinstance(v, dict) and "gflops_per_forward" in v}
    t4 = {r: v["gflops_per_forward"] * 1e9 * fwd / (t4_tflops * 1e12) for r, v in deep.items()}
    cpu_deep = {r: v["cpu_s_per_forward"] * fwd for r, v in deep.items()}
    decode = res["decode_all_s_per_img"] * n  # already measured with n_cores workers
    return {"assumed_t4_effective_tflops": t4_tflops, "t4_model_s_one_gpu": t4,
            "t4_model_total_s_one_gpu": sum(t4.values()), "t4_model_total_s_two_gpus": sum(t4.values()) / 2,
            "decode_and_cpu_features_s_4cores": decode,
            "estimated_kaggle_wall_s_two_t4": max(decode, sum(t4.values()) / 2) + 0.2 * min(decode, sum(t4.values()) / 2),
            "cpu_model_total_s": cpu_deep}


# ================================================================================================ data for search
def _label_seed(idx_or_df, fracs=engine.FRACTIONS):
    """Smallest seed whose nested experiment subsets all contain both labels."""
    g, y = idx_or_df.experiment.astype(str).to_numpy(), idx_or_df.good.to_numpy()
    for s in range(1000):
        if all(len(np.unique(y[engine.nested_subset(g, f, seed=s)])) == 2 for f in fracs):
            return s
    raise RuntimeError("no seed gives both labels")


class Data:
    def __init__(self, emb_dir, smoke=False):
        self.dir = Path(emb_dir)
        idx = pd.read_csv(self.dir / "index.csv", dtype={"imageID": str, "experiment": str})
        meta = json.loads((self.dir / "q4_meta.json").read_text()) if (self.dir / "q4_meta.json").exists() else {}
        rows = np.arange(len(idx))
        if smoke and not meta.get("smoke"):
            rows = engine.nested_subset(idx.experiment.to_numpy(), 0.10, seed=_label_seed(idx, (0.10,)))
        self.rows, self.idx = rows, idx.iloc[rows].reset_index(drop=True)
        self.y, self.g = self.idx.good.to_numpy(), self.idx.experiment.to_numpy()
        self.cell, self.day = self.idx.cell.to_numpy(), self.idx.day.to_numpy()
        self.seed = _label_seed(self.idx)
        self.stage = {f: engine.nested_subset(self.g, f, seed=self.seed) for f in engine.FRACTIONS}
        self._X = {}

    def reps(self):
        """Representations with every needed preprocessing on disk."""
        out = []
        for r in P.ALL_REPS:
            preps = ["centre_crop"] if r == "virtual_stain_features" else P.PREPS
            if all(P.emb_file(self.dir, r, p).exists() for p in preps):
                out.append(r)
        return out

    def X(self, rep, prep):
        k = (rep, prep)
        if k not in self._X:
            self._X[k] = np.load(P.emb_file(self.dir, rep, prep), mmap_mode="r")[self.rows].astype(np.float32)
        return self._X[k]

    def pix(self, prep):
        return np.load(P.emb_file(self.dir, "pix", prep), mmap_mode="r")[self.rows]

    def features(self, path):
        X = self.X(path["representation"], path["preprocessing"])
        if path["fusion"] == "concat_with_handcrafted":
            X = np.hstack([X, self.X("handcrafted", path["preprocessing"])])
        elif path["fusion"] == "concat_with_virtual":
            X = np.hstack([X, self.X("virtual_stain_features", "centre_crop")])
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


class Rules:
    """Validity rules for a set of available representations (a class, so it can be pickled to worker processes)."""

    def __init__(self, reps):
        self.reps = set(reps)

    def __call__(self, p):
        if p["representation"] not in self.reps:
            return False
        if p["representation"] == "virtual_stain_features" and p["preprocessing"] != "centre_crop":
            return False   # virtual-stain features exist for the centre crop only
        if p["fusion"] == "concat_with_handcrafted" and (p["representation"] == "handcrafted"
                                                          or "handcrafted" not in self.reps):
            return False
        if p["fusion"] == "concat_with_virtual" and (p["representation"] == "virtual_stain_features"
                                                      or "virtual_stain_features" not in self.reps):
            return False
        return True


def rules_for(reps):
    return Rules(reps)


# ================================================================================================ pipelines
def cv_folds(y, g, n=5):
    """Deterministic 5-fold CV grouped by experiment (stratified on the label), same folds for every path."""
    from sklearn.model_selection import StratifiedGroupKFold
    n = min(n, len(np.unique(g)))
    return list(StratifiedGroupKFold(n, shuffle=True, random_state=0).split(np.zeros(len(y)), y, g))


def _logreg(l1_ratio, C, solver):
    import sklearn
    from sklearn.linear_model import LogisticRegression
    if tuple(int(v) for v in sklearn.__version__.split(".")[:2]) >= (1, 8):
        return LogisticRegression(l1_ratio=l1_ratio, C=C, solver=solver, max_iter=3000)
    pen = {0.0: "l2", 1.0: "l1"}.get(l1_ratio, "elasticnet")
    kw = {"l1_ratio": l1_ratio} if pen == "elasticnet" else {}
    return LogisticRegression(penalty=pen, C=C, solver=solver, max_iter=3000, **kw)


from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402


class CorrFilter(TransformerMixin, BaseEstimator):
    """Drop a feature if |Pearson r| > thr with a feature kept earlier (greedy, column order)."""

    def __init__(self, thr=0.9):
        self.thr = thr

    def fit(self, X, y=None):
        c = np.abs(np.nan_to_num(np.corrcoef(X, rowvar=False)))
        keep = []
        for j in range(X.shape[1]):
            if not keep or c[j, keep].max() <= self.thr:
                keep.append(j)
        self.keep_ = np.array(keep)
        return self

    def transform(self, X):
        return X[:, self.keep_]


class L1Select(CorrFilter):
    """Keep features with non-zero L1-logistic coefficients (C=0.05 on standardised features; >= 5 kept)."""
    def __init__(self, C=0.05, min_keep=5):
        self.C, self.min_keep = C, min_keep

    def fit(self, X, y):
        coef = np.abs(_logreg(1.0, self.C, "liblinear").fit(X, y).coef_[0])
        keep = np.where(coef > 0)[0]
        if len(keep) < self.min_keep:
            keep = np.argsort(-coef)[: self.min_keep]
        self.keep_ = np.sort(keep)
        return self

    def get_params(self, deep=True):
        return {"C": self.C, "min_keep": self.min_keep}


def build(path, n_train, n_feat, proba=False, seed=0):
    """Imputer -> scaler -> reduction -> model. Hyper-parameters are fixed (documented defaults)."""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    red = {"none": [], "pca_50": [PCA(min(50, n_feat, n_train - 1), random_state=seed)],
           "l1_selection": [L1Select()], "correlation_filter_0.9": [CorrFilter(0.9)]}[path["reduction"]]
    m = path["model"]
    clf = {"logistic_l1": lambda: _logreg(1.0, 0.1, "liblinear"),
           "logistic_l2": lambda: _logreg(0.0, 0.1, "lbfgs"),
           "logistic_elasticnet": lambda: _logreg(0.5, 0.1, "saga"),
           "svm": lambda: SVC(C=1.0, gamma="scale", probability=proba, random_state=seed),
           "random_forest": lambda: RandomForestClassifier(300, max_features="sqrt", min_samples_leaf=2,
                                                           n_jobs=1, random_state=seed),
           "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                                                            early_stopping=False, random_state=seed),
           "mlp": lambda: MLPClassifier((256,), alpha=1e-3, early_stopping=True, max_iter=300, random_state=seed),
           "knn": lambda: KNeighborsClassifier(min(15, max(1, n_train // 5)), weights="distance")}[m]()
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), *red, clf)


def _scores(pipe, X):
    if hasattr(pipe, "predict_proba") and (not hasattr(pipe[-1], "probability") or pipe[-1].probability):
        return pipe.predict_proba(X)[:, 1]
    return pipe.decision_function(X)


def oof(path, X, y, g, splits=None, proba=False, fit_fn=None):
    """Out-of-fold predictions; a training fold with one label predicts its prevalence."""
    p = np.zeros(len(y))
    for tr, te in (splits if splits is not None else cv_folds(y, g)):
        if len(np.unique(y[tr])) < 2:
            p[te] = y[tr].mean()
            continue
        if fit_fn is not None:
            p[te] = fit_fn(X[tr], y[tr], X[te])
        else:
            pipe = build(path, len(tr), X.shape[1], proba=proba).fit(X[tr], y[tr])
            p[te] = _scores(pipe, X[te])
    return p


def path_key(path):
    s = "-".join(str(path[k]) for k in sorted(path))
    return hashlib.md5(s.encode()).hexdigest()[:10]


def path_name(path):
    if "trained_layers" in path:
        return f"resnet18_ft/{path['preprocessing']}/{path['trained_layers']}/lr{float(path['lr']):g}"
    return "/".join(str(path[k]) for k in ("preprocessing", "representation", "fusion", "reduction", "model"))


def _auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


# ------------------------------------------------------------------ engine score functions (module level: spawn)
_DATA = {}


def _data():
    if "d" not in _DATA:
        _DATA["d"] = Data(os.environ["Q4_EMB_DIR"], smoke=os.environ.get("Q4_SMOKE") == "1")
    return _DATA["d"]


def _cached(kind, path, frac, compute):
    """OOF predictions are stored per (path, fraction): duplicate proposals and restarts are free."""
    d = Path(os.environ["Q4_OUT"]) / "oof"
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{kind}_{path_key(path)}_{int(frac * 100)}.npz"
    if f.exists():
        return np.load(f)["p"]
    p = compute()
    np.savez(f, p=p, path=json.dumps(path))
    return p


def score_search(path, frac, trial=None):
    from threadpoolctl import threadpool_limits
    d = _data()
    rows = d.stage[frac]

    def compute():
        with threadpool_limits(1):
            return oof(path, d.features(path)[rows], d.y[rows], d.g[rows])
    return _auc(d.y[rows], _cached("search", path, frac, compute))


def score_finetune(path, frac, trial=None):
    d = _data()
    rows = d.stage[frac]
    epochs = int(os.environ.get("Q4_EPOCHS", "8"))
    dev = _ft_device(trial)

    def compute():
        X = d.pix(path["preprocessing"])[rows]
        fit = lambda Xtr, ytr, Xte: ft_fit_predict(Xtr, ytr, Xte, path["trained_layers"], float(path["lr"]),
                                                   epochs, dev)[0]
        return oof(path, X, d.y[rows], d.g[rows], fit_fn=fit)
    return _auc(d.y[rows], _cached("finetune", path, frac, compute))


# ================================================================================================ fine-tuning
def _ft_device(trial=None):
    import torch
    n = torch.cuda.device_count()
    if not n:
        torch.set_num_threads(int(os.environ.get("Q4_THREADS", "4")))
        return "cpu"
    return f"cuda:{(trial.number if trial is not None else 0) % n}"


def _to_input(xb, dev):
    import torch
    x = xb.to(dev, non_blocking=True).float().div_(255)[:, None].expand(-1, 3, -1, -1)
    m = torch.tensor((0.485, 0.456, 0.406), device=dev).view(1, 3, 1, 1)
    s = torch.tensor((0.229, 0.224, 0.225), device=dev).view(1, 3, 1, 1)
    return (x - m) / s


def _augment(x):
    """Random h/v flips and a small rotation (+-15 deg, reflection padding) per sample."""
    import torch
    import torch.nn.functional as F
    B = x.shape[0]
    x = torch.where((torch.rand(B, device=x.device) < 0.5).view(-1, 1, 1, 1), x.flip(3), x)
    x = torch.where((torch.rand(B, device=x.device) < 0.5).view(-1, 1, 1, 1), x.flip(2), x)
    a = (torch.rand(B, device=x.device) * 2 - 1) * np.pi / 12
    r = x.shape[-2] / x.shape[-1]
    z = torch.zeros_like(a)
    theta = torch.stack([torch.stack([a.cos(), -a.sin() * r, z], 1), torch.stack([a.sin() / r, a.cos(), z], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)


def _forward(m, x, layers):
    import torch
    with torch.set_grad_enabled(layers == "all_layers" and m.training):
        h = m.maxpool(m.act1(m.bn1(m.conv1(x))))
        h = m.layer3(m.layer2(m.layer1(h)))
    with torch.set_grad_enabled(layers != "last_layer" and m.training):
        h = m.layer4(h)
    return m.fc(m.global_pool(h)).squeeze(1)


def ft_fit_predict(Xtr, ytr, Xte, layers, lr, epochs, dev, seed=0, bs=32, return_model=False):
    """ResNet18 (ImageNet) fine-tuned on grey pixels (n, T, H, W). Tiles are separate training samples with the
    image label; the image prediction is the mean tile probability. fp16 autocast + GradScaler on CUDA."""
    import timm
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    np.random.seed(seed)
    T = Xtr.shape[1]
    xtr = torch.from_numpy(np.ascontiguousarray(Xtr).reshape(-1, *Xtr.shape[2:]))
    ytr_t = torch.from_numpy(np.repeat(ytr, T).astype(np.float32))
    m = timm.create_model("resnet18.a1_in1k", pretrained=True, num_classes=1).to(dev)
    trainable = {"last_layer": [m.fc], "last_block": [m.layer4, m.fc], "all_layers": [m]}[layers]
    for q in m.parameters():
        q.requires_grad = False
    params = [q for mod in trainable for q in mod.parameters()]
    for q in params:
        q.requires_grad = True
    cuda = dev.startswith("cuda")
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    steps = max(1, epochs * int(np.ceil(len(xtr) / bs)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=cuda)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        m.train()
        if layers != "all_layers":  # frozen BN keeps its statistics
            for mod in m.modules():
                mod.train(False)
            for mod in trainable:
                mod.train(True)
        perm = torch.randperm(len(xtr), generator=gen)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", torch.float16, enabled=cuda):
                loss = F.binary_cross_entropy_with_logits(_forward(m, _augment(_to_input(xtr[b], dev)), layers).float(),
                                                          ytr_t[b].to(dev))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
    m.eval()
    xte, out = torch.from_numpy(np.ascontiguousarray(Xte).reshape(-1, *Xte.shape[2:])), []
    with torch.no_grad(), torch.autocast("cuda", torch.float16, enabled=cuda):
        for i in range(0, len(xte), 64):
            out.append(torch.sigmoid(_forward(m, _to_input(xte[i:i + 64], dev), layers).float()).cpu().numpy())
    p = np.concatenate(out).reshape(len(Xte), T).mean(1)
    return p, (m if return_model else None)


# ================================================================================================ search / finetune
def _env(a, extra=None):
    os.environ.update({"Q4_EMB_DIR": str(a.emb_dir), "Q4_OUT": str(a.out), "Q4_SMOKE": "1" if a.smoke else "0",
                       **(extra or {})})


def cmd_search(a):
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"   # one core per engine worker (children inherit this before importing numpy)
    _env(a)
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    d = _data()
    reps = d.reps()
    tree = {"scope": ["all_cell_types"], "preprocessing": P.PREPS, "representation": reps,
            "fusion": ["single", "concat_with_handcrafted", "concat_with_virtual"],
            "reduction": ["none", "pca_50", "l1_selection", "correlation_filter_0.9"], "model": MODELS}
    if "virtual_stain_features" not in reps:
        tree["fusion"].remove("concat_with_virtual")
    print(f"search: {len(d.y)} images, {len(np.unique(d.g))} experiments, label-seed {d.seed}; representations {reps}",
          flush=True)
    s = engine.Search("q4_search", tree, a.out, budget=a.budget, rules=rules_for(reps), baseline=BASELINE)
    t0 = time.time()
    summ = s.run(score_search, workers=a.workers, hours=a.hours)
    summ["wall_s"] = round(time.time() - t0, 1)
    _timing(Path(a.out) / "q4_search_paths.csv", summ, a.workers, budget_full=1000)
    (Path(a.out) / "q4_search_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({k: summ[k] for k in ("evaluated", "completed_100pct", "excluded_options", "wall_s", "timing")},
                     indent=1, default=str))
    print("best:", summ["best"][0] if summ["best"] else None)


def _timing(csv, summ, workers, budget_full):
    """Per-path compute (sum over the stages reached) and an extrapolation to the full budget."""
    df = pd.read_csv(csv)
    df = df[df.skipped.isna() & df.state.isin(["COMPLETE", "PRUNED"])]
    sec = [c for c in df.columns if c.startswith("seconds_")]
    per_path = df[sec].max(1)
    tm = {"mean_s_per_path": float(per_path.mean()), "median_s_per_path": float(per_path.median()),
          "max_s_per_path": float(per_path.max()), "n": int(len(df))}
    if "model" in df:
        tm["mean_s_by_model"] = per_path.groupby(df.model).mean().round(2).to_dict()
        tm["mean_s_by_representation"] = per_path.groupby(df.representation).mean().round(2).to_dict()
    tm[f"extrapolated_h_{budget_full}_paths_{workers}_workers"] = float(per_path.mean() * budget_full / workers / 3600)
    summ["timing"] = tm


def cmd_finetune(a):
    _env(a, {"Q4_EPOCHS": str(a.epochs), "Q4_THREADS": str(a.workers)})
    import optuna
    import torch
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    d = _data()
    tree = {"preprocessing": P.PREPS, "trained_layers": ["last_layer", "last_block", "all_layers"], "lr": [1e-4, 3e-4]}
    workers = a.workers if torch.cuda.device_count() == 0 else max(1, torch.cuda.device_count())
    if torch.cuda.device_count() == 0:
        workers = 1   # CPU: one process using all threads
    s = engine.Search("q4_finetune", tree, a.out, budget=a.budget, baseline=FT_BASELINE)
    t0 = time.time()
    summ = s.run(score_finetune, workers=workers, hours=a.hours)
    summ["wall_s"] = round(time.time() - t0, 1)
    _timing(Path(a.out) / "q4_finetune_paths.csv", summ, workers, budget_full=24)
    df = pd.read_csv(Path(a.out) / "q4_finetune_paths.csv")
    done = df[df.state == "COMPLETE"].sort_values("value", ascending=False)
    if len(done):  # refit the best configuration on all data (for Grad-CAM in assess)
        best = {k: done.iloc[0][k] for k in tree}
        best["lr"] = float(best["lr"])
        t1 = time.time()
        X = d.pix(best["preprocessing"])
        _, m = ft_fit_predict(X, d.y, X[:2], best["trained_layers"], best["lr"], a.epochs, _ft_device(), return_model=True)
        torch.save({"state_dict": m.state_dict(), "path": best, "cv_auc": float(done.iloc[0]["value"]),
                    "epochs": a.epochs}, Path(a.out) / "best_resnet18.pt")
        summ["final_refit_s"] = round(time.time() - t1, 1)
        summ["best_path"] = best
    (Path(a.out) / "q4_finetune_summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps({k: summ.get(k) for k in ("evaluated", "completed_100pct", "best_path", "wall_s", "timing",
                                                "final_refit_s")}, indent=1, default=str))


# ================================================================================================ assess
def laplace_logistic(Xtr, ytr, Xte, prior_sd=1.0):
    """Bayesian logistic regression, Gaussian prior N(0, prior_sd^2) on standardised features; Laplace
    approximation at the MAP; predictive = probit approximation of E[sigmoid(x'w)] (MacKay 1992).
    Returns (mean prob, predictive sd of the logit)."""
    from scipy.optimize import minimize
    A = np.hstack([np.ones((len(Xtr), 1)), Xtr])
    prec = np.r_[1e-2, np.full(Xtr.shape[1], 1 / prior_sd ** 2)]

    def nlp(w):
        z = A @ w
        return np.sum(np.logaddexp(0, z) - ytr * z) + 0.5 * np.sum(prec * w ** 2), A.T @ (1 / (1 + np.exp(-z)) - ytr) + prec * w
    w = minimize(nlp, np.zeros(A.shape[1]), jac=True, method="L-BFGS-B").x
    s = 1 / (1 + np.exp(-(A @ w)))
    H = (A * (s * (1 - s))[:, None]).T @ A + np.diag(prec)
    S = np.linalg.inv(H)
    B = np.hstack([np.ones((len(Xte), 1)), Xte])
    mu, var = B @ w, np.einsum("ij,jk,ik->i", B, S, B)
    return 1 / (1 + np.exp(-mu / np.sqrt(1 + np.pi * var / 8))), np.sqrt(var)


def pymc_logistic(Xtr, ytr, Xte, prior_sd=1.0, draws=500):
    import pymc as pm
    with pm.Model():
        b0 = pm.Normal("b0", 0, 10)
        w = pm.Normal("w", 0, prior_sd, shape=Xtr.shape[1])
        pm.Bernoulli("y", logit_p=b0 + pm.math.dot(Xtr, w), observed=ytr)
        tr = pm.sample(draws, tune=draws, chains=2, cores=1, progressbar=False, random_seed=0)
    W = tr.posterior["w"].values.reshape(-1, Xtr.shape[1])
    B0 = tr.posterior["b0"].values.reshape(-1)
    z = Xte @ W.T + B0
    return (1 / (1 + np.exp(-z))).mean(1), z.std(1)


def bayes_design(path, X, tr, y, max_dim=20):
    """Path's own imputation, scaling and reduction (fit on the training fold), then PCA to <= 20 dims."""
    from sklearn.decomposition import PCA
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    pipe = build(path, len(tr), X.shape[1])[:-1]
    Z = pipe.fit_transform(X[tr], y[tr])
    post = make_pipeline(PCA(min(max_dim, Z.shape[1], len(tr) - 1), random_state=0), StandardScaler()).fit(Z)
    return lambda M: post.transform(pipe.transform(M))


def cmd_assess(a):
    from threadpoolctl import threadpool_limits
    _env(a)
    out = Path(a.out)
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    d = _data()
    y, g, cell, day = d.y, d.g, d.cell, d.day
    t0 = time.time()
    sp = pd.read_csv(Path(a.search_dir or out) / "q4_search_paths.csv")
    keys = ["scope", "preprocessing", "representation", "fusion", "reduction", "model"]
    done = sp[sp.state == "COMPLETE"].sort_values("value", ascending=False).drop_duplicates(keys)
    top = [dict(zip(keys, r)) for r in done[keys].itertuples(index=False)][:4]
    paths = {path_name(p): p for p in top}
    paths.setdefault(path_name(BASELINE), BASELINE)
    base_name, best_name = path_name(BASELINE), path_name(top[0]) if top else path_name(BASELINE)
    splits = cv_folds(y, g)
    preds, reports = {}, {}
    with threadpool_limits(a.workers):
        for n, p in paths.items():  # recomputed with probability outputs (SVM: Platt scaling) for calibration
            preds[n] = oof(p, d.features(p), y, g, splits, proba=True)
            reports[n] = stats.classification_report(y, preds[n], groups=g)
    ft = _finetune_result(a, d)
    if ft:
        preds[ft["name"]] = ft["p"]
        reports[ft["name"]] = stats.classification_report(y, ft["p"], groups=g)
    best_path = paths[best_name]

    def per_cell(p):
        res = {}
        for c in np.unique(cell):
            m = cell == c
            if len(np.unique(y[m])) == 2:
                res[c] = {**stats.bootstrap_ci(_auc, y[m], p[m], groups=g[m], n=1000), "n": int(m.sum()),
                          "n_experiments": int(len(np.unique(g[m]))), "prevalence": float(y[m].mean())}
        return res

    def leave_one_out(path, groups):
        X = d.features(path)
        p = np.zeros(len(y))
        for v in np.unique(groups):
            te, tr = np.where(groups == v)[0], np.where(groups != v)[0]
            p[te] = oof(path, X, y, g, [(tr, te)], proba=True)[te]
        per = {str(v): _auc(y[groups == v], p[groups == v]) for v in np.unique(groups)
               if len(np.unique(y[groups == v])) == 2}
        return {"pooled_auc": _auc(y, p), "per_group_auc": per, "mean_auc": float(np.nanmean(list(per.values())))
                if per else None, "n_groups": int(len(np.unique(groups)))}

    def per_cell_scope(path):
        """scope = per_cell_type: a separate model per cell type, 5-fold CV grouped by experiment inside it."""
        X = d.features(path)
        p = np.full(len(y), np.nan)
        for c in np.unique(cell):
            m = np.where(cell == c)[0]
            if len(np.unique(y[m])) < 2 or len(np.unique(g[m])) < 2:
                p[m] = y[np.where(cell != c)[0]].mean()
                continue
            p[m] = oof(path, X[m], y[m], g[m], proba=True)
        return p

    with threadpool_limits(a.workers):
        p_scope = per_cell_scope(best_path)
        loco = {n: leave_one_out(paths[n], cell) for n in (best_name, base_name)}
        lodo = {n: leave_one_out(paths[n], day) for n in (best_name, base_name)}
    p_cell = np.zeros(len(y))   # reference: knows only the cell type (training-fold prevalence)
    for tr, te in splits:
        prev = pd.Series(y[tr]).groupby(cell[tr]).mean()
        p_cell[te] = pd.Series(cell[te]).map(prev).fillna(y[tr].mean()).to_numpy()

    bayes = _bayes(a, d, top, splits)
    maps = _maps(a, d, ft, best_path, preds[best_name], reports[best_name]["auc"]["estimate"], fig_dir)
    rep = {
        "n_images": int(len(y)), "n_experiments": int(len(np.unique(g))), "smoke": a.smoke,
        "baseline": base_name, "best": best_name, "best_path": best_path, "top4": list(paths)[:4],
        "reports": reports,
        "delong_best_vs_baseline": {**stats.delong(y, preds[best_name], preds[base_name]),
                                    "note": "DeLong assumes independent images; images cluster within experiments, so "
                                            "the p-value is optimistic. Use the cluster-bootstrap CIs."},
        "per_cell_type_auc": {"best": per_cell(preds[best_name]), "baseline": per_cell(preds[base_name])},
        "scope_per_cell_type": {"path": best_name, "auc": _auc(y, p_scope),
                                "per_cell": per_cell(p_scope),
                                "auc_all_cell_types": reports[best_name]["auc"]["estimate"],
                                "note": "scope was not searched (only all_cell_types); evaluated here for the best path"},
        "leave_one_cell_type_out": loco,
        "leave_one_day_out": {**lodo, "note": "experiments rarely span days, so this is close to leaving experiments "
                                              "out by day"},
        "cell_type_only_reference_auc": _auc(y, p_cell),
        "bayesian_logistic": bayes, "maps": maps,
        "finetune": {k: v for k, v in (ft or {}).items() if k != "p"},
        "runtime_s": round(time.time() - t0, 1),
    }
    (out / "q4_assess.json").write_text(json.dumps(rep, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    pd.DataFrame({"imageID": d.idx.imageID, "experiment": g, "cell": cell, "day": day, "good": y,
                  **{f"pred_{n}": p for n, p in preds.items()}}).to_csv(out / "q4_oof_best.csv", index=False)
    rows = [{"name": n, "auc": r["auc"]["estimate"], "lo": r["auc"]["ci_low"], "hi": r["auc"]["ci_high"],
             "baseline": n == base_name} for n, r in reports.items()]
    fig_auc(rows, fig_dir / "fig_q4_auc.png")
    fig_convergence(sp, fig_dir / "fig_q4_convergence.png")
    print(json.dumps({"best": best_name, "auc": reports[best_name]["auc"], "baseline_auc": reports[base_name]["auc"],
                      "delong_p": rep["delong_best_vs_baseline"]["p"], "bayes": {k: v.get("auc") for k, v in bayes.items()
                                                                                   if isinstance(v, dict)},
                      "maps": maps, "runtime_s": rep["runtime_s"]}, indent=1, default=str))


def _finetune_result(a, d):
    fd = Path(a.finetune_dir or Path(a.out).parent / "finetune")
    f = fd / "q4_finetune_paths.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    done = df[df.state == "COMPLETE"].sort_values("value", ascending=False)
    if not len(done):
        return None
    best = {k: done.iloc[0][k] for k in ("preprocessing", "trained_layers", "lr")}
    best["lr"] = float(best["lr"])
    o = fd / "oof" / f"finetune_{path_key(best)}_100.npz"
    if not o.exists():
        return None
    return {"name": path_name(best), "path": best, "p": np.load(o)["p"], "cv_auc": float(done.iloc[0]["value"]),
            "model_file": str(fd / "best_resnet18.pt") if (fd / "best_resnet18.pt").exists() else None}


def _bayes(a, d, top, splits):
    """Bayesian logistic on each of the best 4 pipelines' features (the path's reduction, then PCA <= 20)."""
    method = a.bayes
    if method == "auto":
        try:
            import pymc  # noqa: F401
            method = "laplace" if a.smoke else "pymc"
        except ImportError:
            method = "laplace"
    fit = pymc_logistic if method == "pymc" else laplace_logistic
    out = {"method": method, "note": "Gaussian prior N(0,1) on standardised PCA (<=20) components of the path's "
                                     "reduced features; Laplace = MAP + Hessian, probit predictive approximation"}
    for p in top:
        X = d.features(p)
        prob, sd = np.zeros(len(d.y)), np.zeros(len(d.y))
        for tr, te in splits:
            tf = bayes_design(p, X, tr, d.y)
            prob[te], sd[te] = fit(tf(X[tr]), d.y[tr], tf(X[te]))
        out[path_name(p)] = {"auc": _auc(d.y, prob), "brier": float(np.mean((prob - d.y) ** 2)),
                             "mean_logit_sd": float(sd.mean()),
                             "logit_sd_wrong_vs_right": [float(sd[(prob > .5) != d.y].mean()),
                                                         float(sd[(prob > .5) == d.y].mean())]}
    return out


def _pick6(y, p):
    """2 correct + 1 incorrect per true label, most confident first."""
    pick = []
    for lab in (1, 0):
        conf = p if lab == 1 else 1 - p
        right = np.where((y == lab) & (conf >= 0.5))[0]
        wrong = np.where((y == lab) & (conf < 0.5))[0]
        pick += [(i, lab, True) for i in right[np.argsort(-conf[right])][:2]]
        pick += [(i, lab, False) for i in wrong[np.argsort(conf[wrong])][:1]]
    return pick


def _maps(a, d, ft, best_path, p_best, best_auc, fig_dir):
    """Grad-CAM of the fine-tuned ResNet18 if competitive (AUC >= best search AUC - 0.02), else DINOv2
    patch-to-CLS cosine-similarity maps. 6 chips: good/bad x correct/incorrect."""
    import torch
    prep = best_path["preprocessing"]
    try:
        if ft and ft.get("model_file") and ft["cv_auc"] >= best_auc - 0.02:
            import timm
            ck = torch.load(ft["model_file"], map_location="cpu")
            m = timm.create_model("resnet18", pretrained=False, num_classes=1)
            m.load_state_dict(ck["state_dict"])
            m.eval()
            prep, p_sel, kind = ck["path"]["preprocessing"], ft["p"], "gradcam_resnet18_finetuned"

            def heat(img):
                x = _to_input(torch.from_numpy(img[None]), "cpu").requires_grad_(False)
                feats = {}
                h = m.layer4.register_forward_hook(lambda mod, i, o: feats.__setitem__("a", o))
                out = m(x)[0, 0]
                a_ = feats["a"]
                gr = torch.autograd.grad(out, a_)[0]
                h.remove()
                cam = torch.relu((gr.mean((2, 3), keepdim=True) * a_).sum(1))[0].detach().numpy()
                return cam
        else:
            import timm
            rep = best_path["representation"] if best_path["representation"].startswith("dinov2") else "dinov2_small"
            m = timm.create_model(P.DEEP[rep][1], pretrained=True, num_classes=0, dynamic_img_size=True).eval()
            p_sel, kind = p_best, f"patch_similarity_{rep}"

            def heat(img):
                W = img.shape[1] // 14 * 14
                x = _to_input(torch.from_numpy(np.ascontiguousarray(img[None, :, :W])), "cpu")
                with torch.no_grad():
                    t = m.forward_features(x)[0]
                cls, pt = t[0], t[m.num_prefix_tokens:]
                sim = torch.nn.functional.cosine_similarity(pt, cls[None], dim=1)
                return sim.reshape(img.shape[0] // 14, W // 14).numpy()
        pix = d.pix(prep)
        p_show = p_sel if p_sel.min() >= 0 and p_sel.max() <= 1 else 1 / (1 + np.exp(-p_sel))
        pick = _pick6(d.y, p_show)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 3, figsize=(12, 6.4), facecolor=BG)
        for ax in axes.ravel():
            ax.axis("off")
        for k, (i, lab, ok) in enumerate(pick):
            tiles = pix[i]
            imgs = [tiles[t] for t in range(len(tiles))]
            hm = [heat(t_) for t_ in imgs]
            if len(imgs) == 4:  # 2 x 2 mosaic for tiles_mean
                img = np.block([[imgs[0], imgs[1]], [imgs[2], imgs[3]]])
                hm = [np.block([[hm[0], hm[1]], [hm[2], hm[3]]])]
            else:
                img = imgs[0]
            ax = axes[0 if lab == 1 else 1, k % 3]
            ax.imshow(img, cmap="gray")
            ax.imshow(hm[0], cmap="magma", alpha=0.45, extent=(0, img.shape[1], img.shape[0], 0))
            ax.set_title(f"true {'good' if lab else 'bad'}, {'correct' if ok else 'incorrect'} "
                         f"(p_good={p_show[i]:.2f})\n{d.idx.imageID[i]} {d.cell[i]}",
                         fontsize=8, color=INK)
        fig.suptitle(f"Q4 maps ({kind}, preprocessing {prep})", x=0.01, ha="left", fontsize=11, color=INK)
        fig.tight_layout()
        f = fig_dir / "fig_q4_maps.png"
        fig.savefig(f, dpi=130, facecolor=BG)
        plt.close(fig)
        return {"kind": kind, "file": str(f), "images": [d.idx.imageID[i] for i, _, _ in pick]}
    except Exception as e:  # noqa: BLE001 (maps are optional: no weights offline, no pixels, ...)
        return {"kind": "skipped", "error": f"{type(e).__name__}: {e}"}


# ================================================================================================ figures
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=INK, labelsize=8)
    ax.set_facecolor(BG)


def fig_auc(rows, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = sorted(rows, key=lambda r: (not r["baseline"], -r["auc"]))
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(rows) + 1.6), facecolor=BG)
    _style(ax)
    ypos = np.arange(len(rows))[::-1]
    for yy, r in zip(ypos, rows):
        ax.barh(yy, r["auc"] - 0.5, left=0.5, height=0.6, color=GREY if r["baseline"] else ORANGE)
        ax.errorbar(r["auc"], yy, xerr=[[r["auc"] - r["lo"]], [r["hi"] - r["auc"]]], color=INK, lw=1, capsize=3)
        ax.text(r["hi"] + 0.006, yy, f"{r['auc']:.3f} [{r['lo']:.2f}, {r['hi']:.2f}]", va="center", fontsize=8, color=INK)
    ax.set_yticks(ypos, [r["name"] + (" (baseline)" if r["baseline"] else "") for r in rows], fontsize=8)
    ax.set_xlim(0.45, min(1.0, max(r["hi"] for r in rows) + 0.12))
    ax.set_xlabel("ROC AUC (out-of-fold, 5-fold CV grouped by experiment; 95% CI resampling experiments)", color=INK)
    ax.set_title("Q4 chip quality control: best paths", loc="left", fontsize=11, color=INK)
    ax.grid(axis="x", color="#e8e7e2", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)


def fig_convergence(sp, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ev = sp[sp.skipped.isna() & sp.state.isin(["COMPLETE", "PRUNED"])].sort_values("trial").reset_index(drop=True)
    x = np.arange(1, len(ev) + 1)
    fig, ax = plt.subplots(figsize=(8, 4), facecolor=BG)
    _style(ax)
    s25 = ev.get("score_25", pd.Series(np.nan, index=ev.index))
    ax.scatter(x, s25, s=10, color=GREY, label="AUC at 25% of experiments", zorder=2)
    full = ev.value.where(ev.state == "COMPLETE")
    ax.scatter(x, full, s=14, color=ORANGE, label="AUC at 100% (completed paths)", zorder=3)
    ax.plot(x, full.cummax().ffill(), color=INK, lw=2, label="best so far (100%)", zorder=4)
    base = ev[(ev.preprocessing == BASELINE["preprocessing"]) & (ev.representation == BASELINE["representation"]) &
              (ev.fusion == "single") & (ev.reduction == "none") & (ev.model == BASELINE["model"])]
    if len(base) and pd.notna(base.value.iloc[0]):
        ax.axhline(base.value.iloc[0], color=GREY, lw=1, ls="--", label="baseline (100%)")
    ax.set_xlabel("paths evaluated (warm-up first, then TPE)", color=INK)
    ax.set_ylabel("out-of-fold ROC AUC", color=INK)
    ax.set_title("Q4 search convergence", loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="y", color="#e8e7e2", lw=0.6)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=BG)
    plt.close(fig)


# ================================================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["embed", "search", "finetune", "assess"])
    ap.add_argument("--zip", help="OOC_image_dataset.zip (default: found under /kaggle/input)")
    ap.add_argument("--xlsx", help="OOC_datasheet.xlsx (default: found under /kaggle/input, else local data)")
    ap.add_argument("--img-dir", help="extracted image folder (local default when no --zip)")
    ap.add_argument("--virt-csv", help="chip_features.csv with virt_* columns")
    ap.add_argument("--emb-dir")
    ap.add_argument("--out")
    ap.add_argument("--search-dir", help="assess: search output dir (default --out)")
    ap.add_argument("--finetune-dir", help="assess: finetune output dir (default <out>/../finetune)")
    ap.add_argument("--reps", nargs="*", help="embed: representations (default all; smoke: 3)")
    ap.add_argument("--budget", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hours", type=float, default=10.0)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--bench", type=int, default=0, help="embed: only time N images per representation")
    ap.add_argument("--bayes", default="auto", choices=["auto", "laplace", "pymc"])
    ap.add_argument("--smoke", action="store_true", help="3 representations, budget 12, 10%% of experiments, 1 epoch")
    a = ap.parse_args(argv)
    base = Path("/kaggle/working/q4") if is_kaggle() else OUT / ("smoke" if a.smoke else "v2")
    if not a.emb_dir:  # Kaggle CPU session: the embeddings come in as an input dataset
        found = find_input("q4_meta.json") if a.command != "embed" else None
        a.emb_dir = str(Path(found).parent) if found else str(base / "emb")
    a.out = a.out or str(base / {"embed": "emb", "search": "search", "finetune": "finetune", "assess": "assess"}[a.command])
    a.budget = a.budget or (12 if a.smoke else {"search": 1000, "finetune": 24}.get(a.command, 0))
    a.epochs = a.epochs or (1 if a.smoke else 8)
    if a.command == "assess" and not a.search_dir:
        a.search_dir = str(Path(a.out).parent / "search")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    {"embed": cmd_embed, "search": cmd_search, "finetune": cmd_finetune, "assess": cmd_assess}[a.command](a)


if __name__ == "__main__":
    main()
