"""Evaluate virtual staining on the unseen test plates, at three levels:

1. Pixels:     per-channel Pearson r / SSIM / PSNR of virtual vs real stains.
2. Toxicity:   nuclei count per well (a standard cytotoxicity readout) from virtual vs real DNA stain.
3. Phenotype:  well-level image profiles from REAL stains, VIRTUAL stains, and BRIGHT-FIELD only, scored by
               - phenotypic activity mAP: does a compound's profile retrieve its replicate on the other
                 plate ahead of DMSO wells? (following the copairs / JUMP evaluation protocol)
               - fraction of compounds significantly retrieved (permutation null, p < 0.05)

All metrics are reported separately for compounds held out from training ("test" split) and seen ones.

    python evaluate.py --run base
"""
import argparse
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity

import config as C
from data import load_model, load_plate, norm_params, normalise, predict

TGT = [C.CHANNELS[c] for c in C.TARGET_CH]
BF = [C.CHANNELS[c] for c in C.INPUT_CH]


# ---------------------------------------------------------------------------------------------- features
def channel_features(img):
    """Hand-crafted, CellProfiler-flavoured per-channel image features on a normalised (H, W) field."""
    f = {}
    q = np.percentile(img, [10, 50, 90, 99])
    f.update(mean=img.mean(), std=img.std(), p10=q[0], p50=q[1], p90=q[2], p99=q[3])
    try:
        f["frac_fg"] = (img > threshold_otsu(img)).mean()
    except ValueError:
        f["frac_fg"] = 0.0
    gy, gx = np.gradient(img)
    f["grad"] = np.hypot(gx, gy).mean()
    f["lap"] = np.abs(ndi.laplace(img)).mean()
    prev = img
    for s in (1, 2, 4, 8):  # granularity-like spectrum
        g = ndi.gaussian_filter(img, s)
        f[f"dog{s}"] = np.abs(prev - g).mean()
        prev = g
    return f


def nuclei(dna):
    """Count nuclei as local maxima of the smoothed DNA channel above Otsu. Returns (count, fg area fraction)."""
    sm = ndi.gaussian_filter(dna, 2.5)
    try:
        t = threshold_otsu(sm)
    except ValueError:
        return 0, 0.0
    pk = peak_local_max(sm, min_distance=5, threshold_abs=t, exclude_border=False)
    return len(pk), float((sm > t).mean())


def site_features(chans, names, dna=None):
    f = {}
    for img, n in zip(chans, names):
        for k, v in channel_features(img).items():
            f[f"{n}_{k}"] = v
    small = chans[:, ::4, ::4].reshape(len(chans), -1)
    cc = np.corrcoef(small)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            f[f"corr_{names[i]}_{names[j]}"] = cc[i, j]
    if dna is not None:
        f["n_nuclei"], f["nuc_area"] = nuclei(dna)
    return f


# ---------------------------------------------------------------------------------------------- metrics
def texture_fidelity(wells):
    """Per channel, across wells: how well virtual texture tracks real texture (Pearson r of fine-scale
    features) and how much fine-scale energy survives (median virtual/real ratio; 1.0 = no blur)."""
    out = {}
    for c in TGT:
        r = [np.corrcoef(wells[f"real_{c}_{k}"], wells[f"virt_{c}_{k}"])[0, 1] for k in ("lap", "dog1", "dog2")]
        e = [(wells[f"virt_{c}_{k}"] / wells[f"real_{c}_{k}"]).median() for k in ("lap", "dog1")]
        out[c] = {"texture_r": round(float(np.mean(r)), 3), "fine_energy_ratio": round(float(np.mean(e)), 3),
                  "mean_intensity_r": round(float(np.corrcoef(wells[f"real_{c}_mean"], wells[f"virt_{c}_mean"])[0, 1]), 3)}
    return out


def robust_z(df, feats):
    """Per plate: centre/scale every feature by the DMSO wells' median / MAD."""
    out = []
    for _, g in df.groupby("plate"):
        ctl = g[g.pert_iname == "DMSO"][feats]
        med, mad = ctl.median(), (ctl - ctl.median()).abs().median() * 1.4826 + 1e-6
        g = g.copy()
        g[feats] = ((g[feats] - med) / mad).clip(-50, 50)
        out.append(g)
    return pd.concat(out)


def average_precision(rel):
    hits = np.cumsum(rel)
    return float((hits[rel == 1] / (np.where(rel == 1)[0] + 1)).mean())


def null_ap(n_pos, n_total, n=2000, rng=np.random.default_rng(0)):
    base = np.zeros(n_total)
    base[:n_pos] = 1
    return np.array([average_precision(rng.permutation(base)) for _ in range(n)])


def activity_map(df, feats):
    """Phenotypic activity: each compound well queries replicate wells (same compound, other plate) vs DMSO wells."""
    X = df[feats].to_numpy(np.float64)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    names, plates = df.pert_iname.to_numpy(), df.plate.to_numpy()
    is_ctl = names == "DMSO"
    rows, nulls = [], {}
    for i in np.where(~is_ctl)[0]:
        pos = np.where((names == names[i]) & (plates != plates[i]))[0]
        if len(pos) == 0:
            continue
        neg = np.where(is_ctl)[0]
        cand = np.concatenate([pos, neg])
        sims = X[cand] @ X[i]
        rel = np.r_[np.ones(len(pos)), np.zeros(len(neg))][np.argsort(-sims)]
        ap = average_precision(rel)
        key = (len(pos), len(cand))
        if key not in nulls:
            nulls[key] = null_ap(*key)
        p = (np.sum(nulls[key] >= ap) + 1) / (len(nulls[key]) + 1)
        rows.append({"pert_iname": names[i], "split": df.split.iloc[i], "ap": ap, "p": p})
    r = pd.DataFrame(rows).groupby(["pert_iname", "split"], as_index=False).agg(ap=("ap", "mean"), p=("p", "max"))
    return r


def site_metrics(meta, real, virt, bf):
    """Pixel metrics and features for one field (runs in a worker process)."""
    pix = []
    for c, n in enumerate(TGT):
        rng = max(real[c].max() - real[c].min(), 1e-6)
        pix.append({**meta, "channel": n,
                    "pearson": np.corrcoef(virt[c].ravel(), real[c].ravel())[0, 1],
                    "ssim": structural_similarity(real[c], virt[c], data_range=rng),
                    "psnr": 10 * np.log10(rng ** 2 / np.mean((real[c] - virt[c]) ** 2))})
    s = {**meta}
    s.update({f"real_{k}": v for k, v in site_features(real, TGT, real[0]).items()})
    s.update({f"virt_{k}": v for k, v in site_features(virt, TGT, virt[0]).items()})
    s.update({f"bf_{k}": v for k, v in site_features(bf, BF).items()})
    return pix, s


# ---------------------------------------------------------------------------------------------- main
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="base", help="a run folder in $VCP_RUNS (legacy runs: VCP_RUNS=legacy/runs)")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--plates", nargs="+", help="default: config.TEST_PLATES")
    ap.add_argument("--tag", default="", help="write to runs/<run>/eval_<tag>/ instead of runs/<run>/")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--winner", help="q1_winner.json: evaluate its seed models with its chosen inference mode; "
                                     "results go to --out")
    ap.add_argument("--out", help="output folder for --winner")
    ap.add_argument("--save-virtual", help="also save each virtual field (float16) as <dir>/<plate>_<field>.npy, the Q3 cache format")
    ap.add_argument("--limit", type=int, default=0, help=argparse.SUPPRESS)  # smoke tests: first N fields per plate
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.winner:
        from data import predict_multi
        w = json.loads(Path(args.winner).read_text())
        ms = [load_model(Path(d), "best.pt", dev) for d in w["run_dirs"]]
        cfg = ms[0][1]
        models = [m for m, _ in ms] if w["inference"].startswith("seed_ensemble") else [ms[0][0]]
        tta = w["inference"].endswith("tta")
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "train_log.json").write_text(json.dumps({"args": cfg, "winner": w}, indent=1))

        def infer(x):
            return predict_multi(models, cfg, x, dev, tta)
    else:
        out = C.RUNS / args.run
        model, cfg = load_model(out, args.ckpt, dev)

        def infer(x):
            return predict(model, cfg, x, dev)

    plates = args.plates or C.TEST_PLATES
    if args.tag:
        out = out / f"eval_{args.tag}"
        out.mkdir(exist_ok=True)
    pix, sites, futs = [], [], []

    def drain(keep):
        # Bounded queue: each pending field holds ~15 MB of arrays.
        while len(futs) > keep:
            p, s = futs.pop(0).result()
            pix.extend(p)
            sites.append(s)
            if len(sites) % 200 == 0:
                print(f"  {len(sites)} sites done", flush=True)

    with ProcessPoolExecutor(args.workers) as pool:
        for plate in plates:
            arr, idx = load_plate(plate)
            lo, hi = norm_params(plate)
            for i in range(min(len(arr), args.limit) if args.limit else len(arr)):
                x = normalise(np.asarray(arr[i]), lo, hi)
                real, bf = x[C.TARGET_CH], x[C.INPUT_CH]
                virt = infer(x)
                if args.save_virtual:
                    Path(args.save_virtual).mkdir(parents=True, exist_ok=True)
                    np.save(Path(args.save_virtual) / f"{plate}_{i:04d}.npy", virt.astype(np.float16))
                meta = idx.iloc[i][["plate", "well", "site", "pert_iname", "split"]].to_dict()
                futs.append(pool.submit(site_metrics, meta, real, virt, bf))
                drain(4 * args.workers)
                if i == 0:
                    np.savez_compressed(out / f"example_{plate}.npz", real=real, virt=virt, bf=bf, **{k: str(v) for k, v in meta.items()})
        drain(0)

    pix = pd.DataFrame(pix)
    sites = pd.DataFrame(sites)
    pix.to_csv(out / "pixel_metrics.csv", index=False)
    sites.to_csv(out / "site_features.csv", index=False)

    def grp(s):
        return np.where(s == "control", "DMSO", np.where(s == "test", "held-out cpds", "seen cpds"))

    pix["group"] = grp(pix.split)
    report = {"pixel": pix.groupby(["group", "channel"])[["pearson", "ssim", "psnr"]].mean().round(3).reset_index().to_dict("records")}

    # Well-level profiles
    num = [c for c in sites.columns if c.startswith(("real_", "virt_", "bf_"))]
    wells = sites.groupby(["plate", "well", "pert_iname", "split"], as_index=False)[num].mean()
    wells["group"] = grp(wells.split)

    # 2. Cytotoxicity readout: nuclei per well, virtual vs real
    tox = {}
    for g, d in [("all", wells), *wells.groupby("group")]:
        tox[g] = {"pearson": float(np.corrcoef(d.real_n_nuclei, d.virt_n_nuclei)[0, 1]),
                  "spearman": float(d.real_n_nuclei.rank().corr(d.virt_n_nuclei.rank())), "n_wells": len(d)}
    report["nuclei_count"] = tox
    from sweep import stats as S
    report["nuclei_agreement"] = {g: S.agreement_report(d.real_n_nuclei.to_numpy(), d.virt_n_nuclei.to_numpy())
                                  for g, d in [("all", wells), *wells.groupby("group")]}

    # 3. Phenotypic activity
    act = {}
    for src in (("real", "virt", "bf") if wells.plate.nunique() > 1 else ()):  # replicates live on other plates
        feats = [c for c in num if c.startswith(src + "_")]
        z = robust_z(wells, feats)
        feats = [f for f in feats if np.isfinite(z[f]).all() and z[f].std() > 1e-6]
        r = activity_map(z, feats)
        r.to_csv(out / f"activity_{src}.csv", index=False)
        act[src] = {g: {"mAP": round(float(d.ap.mean()), 3), "frac_active": round(float((d.p < 0.05).mean()), 3), "n": len(d)}
                    for g, d in [("all", r), *r.groupby(np.where(r.split == "test", "held-out cpds", "seen cpds"))]}
    report["phenotypic_activity"] = act or "not possible: one test plate (compound replicates are on other plates)"
    report["texture"] = texture_fidelity(wells)
    print(pd.DataFrame(report["texture"]).T.to_string())

    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(pd.DataFrame(report["pixel"]).to_string(index=False))
    print(json.dumps({"nuclei_count": tox, "phenotypic_activity": act}, indent=1))


if __name__ == "__main__":
    main()
