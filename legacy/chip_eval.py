"""Apply a single-plane virtual-staining model to real organ-on-a-chip bright-field images.

Dataset: "Organ-on-a-Chip (OOC) Image Dataset" (Zenodo 10203721, CC BY 4.0): 3,072 bright-field images of
chips with 6 cell types, each labelled 'good'/'bad' by a biologist, with days in culture.
There is no fluorescence ground truth, so the model is validated indirectly:

1. Quality:  does a classifier on VIRTUAL-stain features predict the expert good/bad label better than the
             same classifier on BRIGHT-FIELD features? (ROC AUC, cross-validated grouped by experiment date)
2. Growth:   does the virtual nuclei count rise with days in culture on good chips? (Spearman)
3. Pictures: a panel of chip images with their virtual stains.

    python legacy/chip_eval.py --run chip1 --scale 1.0
"""
import os as _os, sys as _sys  # legacy script: shared code from the repository root, runs from legacy/runs
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
_os.environ.setdefault("VCP_RUNS", str(_P(__file__).resolve().parent / "runs"))
import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

import config as C
from data import input_spec, load_model, predict, prep_input
from evaluate import nuclei, site_features

OOC = C.DATA.parent / "ooc"
TGT = [C.CHANNELS[c] for c in C.TARGET_CH]
IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def index_images():
    """Walk the extracted dataset and join with the datasheet on imageID (file stem)."""
    files = [p for p in OOC.rglob("*") if p.suffix.lower() in IMG_EXT]
    rows = []
    for p in files:
        parts = [q.lower() for q in p.relative_to(OOC).parts]
        rows.append({"path": str(p), "imageID": p.stem,
                     "zsplit": next((q for q in parts if q in ("train", "val", "test")), None),
                     "label_dir": next((q for q in parts if q in ("good", "bad")), None)})
    idx = pd.DataFrame(rows)
    sheet = pd.read_excel(OOC / "OOC_datasheet.xlsx")
    sheet = sheet.rename(columns={"cell type": "cell", "Decision 1/2 (good/bad)": "decision"})
    sheet["imageID"] = sheet.imageID.astype(str)
    idx = idx.merge(sheet[["imageID", "cell", "day", "decision"]], on="imageID", how="left")
    idx["good"] = np.where(idx.decision.notna(), idx.decision == 1, idx.label_dir == "good").astype(int)
    idx["experiment"] = idx.imageID.str.extract(r"^(\d+)", expand=False).fillna(idx.imageID)
    return idx


def load_bf(path, scale, crop=1024):
    """Grey-scale float image: central crop (crop x 0.75 crop, native px), rescaled so cells have roughly
    the training pixel size (chip cells are ~2x smaller in pixels than in CPJUMP1 at 540 px)."""
    im = Image.open(path).convert("F")
    if crop:
        w, h = crop, int(crop * 0.75)
        l, t = (im.width - w) // 2, (im.height - h) // 2
        im = im.crop((l, t, l + w, t + h))
    if scale != 1.0:
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.BILINEAR)
    return np.asarray(im, np.float32)


def predict_image(model, cfg, bf):
    """Whole image in one pass (2048x1536 needs ~2.3 GB). The U-Net's GroupNorm uses statistics over the
    whole input, so tiling gives visible seams; tiles are only a fallback when memory runs out."""
    try:
        return predict(model, cfg, bf[None])
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        return predict_tiled(model, cfg, bf)


def predict_tiled(model, cfg, bf, tile=1024, pad=128):
    """Overlapping tiles; each keeps its central region. Image-level normalisation is applied once."""
    H, W = bf.shape
    bf = prep_input(bf[None], input_spec(cfg)[1])[0]
    cfg = {**cfg, "bf_norm": "plate"}  # i.e. no further per-tile rescaling
    out = np.zeros((len(TGT), H, W), np.float32)
    step = tile - 2 * pad
    for y in range(0, H, step):
        for x in range(0, W, step):
            y0, x0 = max(0, y - pad), max(0, x - pad)
            y1, x1 = min(H, y + step + pad), min(W, x + step + pad)
            p = predict(model, cfg, bf[None, y0:y1, x0:x1])
            out[:, y:min(H, y + step), x:min(W, x + step)] = p[:, y - y0:y - y0 + min(step, H - y), x - x0:x - x0 + min(step, W - x)]
    return out


def _half(a):
    """2x block mean over the last two axes."""
    H, W = a.shape[-2] // 2 * 2, a.shape[-1] // 2 * 2
    a = a[..., :H, :W]
    return a.reshape(*a.shape[:-2], H // 2, 2, W // 2, 2).mean((-3, -1))


def features(args):
    """Channel features at half the model resolution (the upscale adds no information and quarters the cost);
    nuclei are counted at model resolution, where the counter's size parameters apply."""
    bf, virt = args
    f = {f"virt_{k}": v for k, v in site_features(_half(virt), TGT).items()}
    f["virt_n_nuclei"], f["virt_nuc_area"] = nuclei(virt[0])
    lo, hi = np.percentile(bf, [1, 99])
    f.update({f"bf_{k}": v for k, v in site_features(_half(((bf - lo) / (hi - lo + 1e-6))[None]), ["BF"]).items()})
    return f


def oof_pred(df, cols, seeds=(0, 1, 2)):
    """Out-of-fold probabilities, 5-fold CV grouped by experiment date, averaged over seeds."""
    X, y, g = df[cols].to_numpy(np.float32), df.good.to_numpy(), df.experiment.astype(str).to_numpy()
    n = min(5, len(np.unique(g)))
    pred = np.zeros(len(df))
    for seed in seeds:
        for tr, te in GroupKFold(n).split(X, y, g):
            clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=seed)
            clf.fit(X[tr], y[tr])
            pred[te] += clf.predict_proba(X[te])[:, 1] / len(seeds)
    return pred


def bootstrap_diff(y, g, p_new, p_ref, n=1000, seed=0):
    """AUC(p_new) - AUC(p_ref), resampling whole experiments. Returns mean, 95% CI, P(diff > 0)."""
    rng = np.random.default_rng(seed)
    groups = {e: np.where(g == e)[0] for e in np.unique(g)}
    keys = list(groups)
    d = []
    for _ in range(n):
        ii = np.concatenate([groups[e] for e in rng.choice(keys, len(keys))])
        if len(np.unique(y[ii])) == 2:
            d.append(roc_auc_score(y[ii], p_new[ii]) - roc_auc_score(y[ii], p_ref[ii]))
    d = np.array(d)
    return {"mean": round(float(d.mean()), 3), "ci95": np.percentile(d, [2.5, 97.5]).round(3).tolist(),
            "p_gt_0": round(float((d > 0).mean()), 3)}


def analyse(df, out):
    bf_cols = [c for c in df.columns if c.startswith("bf_")]
    vt_cols = [c for c in df.columns if c.startswith("virt_")]
    res = {"quality_auc": {}, "growth": {}}
    for name, d in [("all cell types", df), *[(c, g) for c, g in df.groupby("cell") if g.experiment.nunique() >= 3]]:
        if d.good.nunique() < 2:
            continue
        p = {k: oof_pred(d, cols) for k, cols in [("bright-field", bf_cols), ("virtual", vt_cols),
                                                     ("bright-field + virtual", bf_cols + vt_cols)]}
        y, g = d.good.to_numpy(), d.experiment.astype(str).to_numpy()
        res["quality_auc"][name] = {"n": len(d), "experiments": int(d.experiment.nunique()),
                                    **{k: round(float(roc_auc_score(y, v)), 3) for k, v in p.items()}}
        if name == "all cell types":
            res["quality_auc"][name]["gain_both_vs_bf"] = bootstrap_diff(y, g, p["bright-field + virtual"], p["bright-field"])
            res["quality_auc"][name]["gain_virtual_vs_bf"] = bootstrap_diff(y, g, p["virtual"], p["bright-field"])
    for cell, d in df[df.good == 1].groupby("cell"):
        if d.day.nunique() >= 3:
            res["growth"][cell] = {"n": len(d), "spearman_nuclei_vs_day": round(float(d.virt_n_nuclei.corr(d.day, method="spearman")), 3),
                                   "median_nuclei_by_day": d.groupby("day").virt_n_nuclei.median().round(1).to_dict()}
    (out / "chip_report.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="chip1")
    ap.add_argument("--scale", type=float, default=2.0, help="resize factor to match training pixel size")
    ap.add_argument("--crop", type=int, default=1024, help="central crop width in native px (0 = whole image)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (quick look)")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--reuse", action="store_true", help="re-run only the analysis on saved chip_features.csv")
    args = ap.parse_args()
    run = C.RUNS / args.run
    out = run / f"chip_s{args.scale:g}" if not args.limit else run / f"chip_s{args.scale:g}_n{args.limit}"
    out.mkdir(exist_ok=True)
    if args.reuse:
        return analyse(pd.read_csv(out / "chip_features.csv"), out)
    model, cfg = load_model(run)
    assert len(cfg.get("inputs") or []) == 1, "chip images are single-plane: use a model trained with --inputs Brightfield"

    idx = index_images()
    print(idx.groupby(["cell", "good"]).size().unstack(), flush=True)
    if args.limit:
        idx = idx.sample(args.limit, random_state=0)
    feats, examples, futs = [], {}, []
    with ProcessPoolExecutor(args.workers) as pool:
        for k, r in enumerate(idx.itertuples()):
            bf = load_bf(r.path, args.scale, args.crop)
            virt = predict_image(model, cfg, bf)
            futs.append(pool.submit(features, (bf, virt)))
            key = (r.cell, r.good)
            if len(examples.get(key, [])) < 2 and r.cell in ("A549", "HPMEC", "CACO"):
                examples.setdefault(key, []).append((r.imageID, r.day, bf, virt))
            if (k + 1) % 200 == 0:
                print(f"  {k + 1}/{len(idx)}", flush=True)
        feats = [f.result() for f in futs]
    df = pd.concat([idx.reset_index(drop=True), pd.DataFrame(feats)], axis=1)
    df.to_csv(out / "chip_features.csv", index=False)
    np.savez_compressed(out / "examples.npz", **{f"{c}_{g}_{i}": np.concatenate([b[None] / max(np.percentile(b, 99), 1e-6), v])
                                                 for (c, g), lst in examples.items() for i, (_, _, b, v) in enumerate(lst)})

    analyse(df, out)


if __name__ == "__main__":
    main()
