"""Q3 single cell: does segmenting and phenotyping virtual stains give the same cells and states as real stains?

Config: sweep/configs/q3_single_cell.yaml. Four paths = {virtual, real} x {nuclei: DNA -> Cellpose nuclei -> k-means,
cyto3: AGP+DNA -> Cellpose cyto3 -> Leiden}. Each virtual path is scored against its real path.

    ~/.venvs/vcp/bin/python -m sweep.q3_single_cell --stage screen|full [--smoke]

screen = 200 stratified fields (50 DMSO + 150 compound, >=40 from held-out compounds), both segmentations.
full   = all 1,536 test-plate fields, only the segmentation(s) chosen by the screen (best virtual-vs-real F1,
         both kept if within 0.02). Virtual predictions, masks and per-field cell tables are cached in
         runs/q3_single_cell/cache so reruns and the full stage reuse the screen's work.
"""
import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops_table
from skimage.segmentation import find_boundaries

import config as C
import data as D
from sweep import stats

SEED = 0
OUT = C.RUNS / "q3_single_cell"
CACHE = OUT / "cache"
SEGS = {"nuclei": {"cluster": "kmeans", "label": "dna_only -> cellpose_nuclei+kmeans"},
        "cyto3": {"cluster": "leiden", "label": "dna+agp -> cellpose_cyto3+leiden"}}
STAINS = [C.CHANNELS[c] for c in C.TARGET_CH]
SHAPE = ["area", "eccentricity", "solidity", "perimeter"]
FEATURES = SHAPE + [f"mean_{s}" for s in STAINS] + [f"integ_{s}" for s in STAINS]
N_CAL = 4  # fields per (segmentation, source) used for Cellpose's automatic diameter estimate
IOU_MIN, F1_TIE, K_MEANS, LEIDEN_RES, N_PCA, N_UMAP = 0.5, 0.02, 6, 1.0, 10, 20000
REAL, VIRT, BG = "#2a78d6", "#eb6834", "#fcfcfb"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#b5b4ad"
CROP = (135, 270)  # offset, size of the example-figure crop
log = logging.getLogger("q3")


# ------------------------------------------------------------------------------------------------ fields
def all_fields():
    out = []
    for p in C.TEST_PLATES:
        _, idx = D.load_plate(p)
        idx = idx.copy()
        idx["field"] = np.arange(len(idx))
        out.append(idx)
    f = pd.concat(out, ignore_index=True)
    f["well_id"] = f.plate + "_" + f.well
    return f


def select_fields(f, stage, smoke):
    rng = np.random.default_rng(SEED)
    if stage == "full":
        sel = f
    else:
        dmso = f[f.pert_iname == "DMSO"]
        comp = f[f.pert_iname != "DMSO"]
        test = comp[comp.split == "test"]
        t = test.iloc[rng.choice(len(test), 40, replace=False)]
        rest = comp.drop(t.index)
        r = rest.iloc[rng.choice(len(rest), 110, replace=False)]
        d = dmso.iloc[rng.choice(len(dmso), 50, replace=False)]
        sel = pd.concat([d, t, r])
    if smoke:
        d, c = sel[sel.pert_iname == "DMSO"], sel[sel.pert_iname != "DMSO"]
        sel = pd.concat([d.iloc[rng.choice(len(d), 3, replace=False)], c.iloc[rng.choice(len(c), 7, replace=False)]])
    return sel.sort_values(["plate", "field"]).reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ models
class Engines:
    """Lazily loaded U-Net and Cellpose models; drop to CPU on CUDA OOM (the GPU is shared)."""

    def __init__(self):
        import torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.unet = self.cp = None
        self.cp = {}

    def _oom(self, e):
        import torch
        msg = str(e).lower()
        if self.dev == "cuda" and any(k in msg for k in ["out of memory", "cublas", "cudnn", "cuda error"]):
            log.warning("CUDA OOM -> falling back to CPU")
            self.dev, self.unet, self.cp = "cpu", None, {}
            torch.cuda.empty_cache()
            return True
        return False

    def predict(self, x):
        while True:
            try:
                if self.unet is None:
                    self.unet = D.load_model(Path(__file__).resolve().parents[1] / "legacy" / "runs" / "base", dev=self.dev)  # legacy Q3 script
                return D.predict(*self.unet, x, dev=self.dev)
            except RuntimeError as e:
                if not self._oom(e):
                    raise

    def segment(self, seg, dna, agp, diameter=None):
        """diameter=None -> Cellpose's own size model (style regression + refinement segmentation)."""
        from cellpose import models
        bs = 4 if self.dev == "cuda" else 32  # CPU: one forward call (to_mkldnn conversion is per call)
        while True:
            try:
                if seg not in self.cp:
                    self.cp[seg] = models.Cellpose(gpu=self.dev == "cuda", model_type=seg)
                if seg == "nuclei":
                    m, _, _, diam = self.cp[seg].eval(dna, diameter=diameter, channels=[0, 0], batch_size=bs)
                else:  # cytoplasm = AGP (channel 1), nucleus = DNA (channel 2)
                    m, _, _, diam = self.cp[seg].eval(np.stack([agp, dna], -1), diameter=diameter, channels=[1, 2],
                                                      channel_axis=-1, batch_size=bs)
                return m.astype(np.uint16), float(np.atleast_1d(diam)[0])
            except RuntimeError as e:
                if not self._oom(e):
                    raise


# ------------------------------------------------------------------------------------------------ per field
def match(real, virt):
    """IoU matrix from the label contingency table, Hungarian assignment, keep pairs with IoU >= 0.5."""
    nr, nv = int(real.max()), int(virt.max())
    if nr == 0 or nv == 0:
        return np.zeros((0, 2), int), np.zeros(0)
    ov = np.bincount(real.ravel().astype(np.int64) * (nv + 1) + virt.ravel(), minlength=(nr + 1) * (nv + 1))
    ov = ov.reshape(nr + 1, nv + 1)
    inter = ov[1:, 1:].astype(float)
    union = ov[1:, :].sum(1)[:, None] + ov[:, 1:].sum(0)[None] - inter
    iou = inter / np.maximum(union, 1)
    r, c = linear_sum_assignment(-iou)
    keep = iou[r, c] >= IOU_MIN
    return np.column_stack([r[keep] + 1, c[keep] + 1]), iou[r[keep], c[keep]]


def cell_table(mask, stains):
    t = regionprops_table(mask.astype(np.int32), intensity_image=np.moveaxis(stains, 0, -1),
                          properties=["label", "area", "eccentricity", "solidity", "perimeter",
                                      "intensity_mean", "centroid"])
    t = pd.DataFrame(t).rename(columns={"centroid-0": "y", "centroid-1": "x"})
    for k, s in enumerate(STAINS):
        t[f"mean_{s}"] = t.pop(f"intensity_mean-{k}")
        t[f"integ_{s}"] = t[f"mean_{s}"] * t.area
    return t


def load_field(row, eng, timing):
    """Real normalised field (8,H,W) and its virtual stains (5,H,W), predictions cached as float16."""
    vpath = CACHE / "virtual" / f"{row.plate}_{row.field:04d}.npy"
    arr, _ = D.load_plate(row.plate)
    lo, hi = D.norm_params(row.plate)
    x = D.normalise(np.asarray(arr[row.field]), lo, hi)
    if vpath.exists():
        virt = np.load(vpath).astype(np.float32)
    else:
        t0 = time.time()
        virt = eng.predict(x).astype(np.float32)
        np.save(vpath, virt.astype(np.float16))
        timing["predict"] += time.time() - t0
    return x, virt


def calibrate_diameters(f, segs, eng, timing):
    """Cellpose's automatic diameter (diameter=None) is estimated on N_CAL fixed fields (from the screen set) per
    segmentation and source, and its median is then used for every field. Running the size model on every field
    costs ~3 network passes (the nuclei refinement pass alone ~3 min/field on the shared CPU)."""
    path = CACHE / "diameters.json"
    cal = json.loads(path.read_text()) if path.exists() else {}
    todo = [s for s in segs if s not in cal]
    if not todo:
        return cal
    rows = select_fields(f, "screen", smoke=True).iloc[:N_CAL]
    for seg in todo:
        est = {"real": [], "virtual": []}
        for row in rows.itertuples():
            x, virt = load_field(row, eng, timing)
            for src, im in {"real": x[C.TARGET_CH], "virtual": virt}.items():
                t0 = time.time()
                est[src].append(eng.segment(seg, im[0], im[3], diameter=None)[1])
                timing["calibrate"] += time.time() - t0
                log.info("  calibrate %s %s %s_%d: diameter %.1f (%.0fs, %s)", seg, src, row.plate, row.field,
                         est[src][-1], time.time() - t0, eng.dev)
        cal[seg] = {src: {"diameter": float(np.median(v)), "per_field": v,
                          "fields": [f"{r.plate}_{r.field}" for r in rows.itertuples()]} for src, v in est.items()}
        log.info("calibrated %s diameters: real %.1f px, virtual %.1f px", seg, cal[seg]["real"]["diameter"],
                 cal[seg]["virtual"]["diameter"])
        path.write_text(json.dumps(cal, indent=2))
    return cal


def process_field(row, segs, eng, timing, cal):
    key = f"{row.plate}_{row.field:04d}"
    x, virt = load_field(row, eng, timing)
    imgs = {"real": x[C.TARGET_CH], "virtual": virt}
    out = []
    for seg in segs:
        cpath = CACHE / "cells" / f"{key}_{seg}.parquet"
        if cpath.exists():
            out.append(pd.read_parquet(cpath))
            continue
        masks, diams = {}, {}
        for src, im in imgs.items():
            mpath = CACHE / "masks" / f"{key}_{seg}_{src}.npz"
            if mpath.exists():
                z = np.load(mpath)
                masks[src], diams[src] = z["mask"], float(z["diam"])
            else:
                t0 = time.time()
                masks[src], diams[src] = eng.segment(seg, im[0], im[3], diameter=cal[seg][src]["diameter"])
                np.savez_compressed(mpath, mask=masks[src], diam=diams[src])
                timing["segment"] += time.time() - t0
        pairs, iou = match(masks["real"], masks["virtual"])
        tabs = []
        for src, other in [("real", 0), ("virtual", 1)]:
            t = cell_table(masks[src], imgs[src])
            m = dict(zip(pairs[:, other], pairs[:, 1 - other]))
            ii = dict(zip(pairs[:, other], iou))
            t["match"] = t.label.map(m).fillna(0).astype(int)
            t["iou"] = t.label.map(ii)
            t["source"], t["diam"] = src, diams[src]
            tabs.append(t)
        t = pd.concat(tabs, ignore_index=True)
        t["seg"] = seg
        for c in ["plate", "field", "well", "site", "well_id", "pert_iname", "split"]:
            t[c] = getattr(row, c)
        t.to_parquet(cpath, index=False)
        out.append(t)
    return out


# ------------------------------------------------------------------------------------------------ evaluation
def seg_metrics(cells, fields):
    """Pooled precision/recall/F1 over all fields (+ well-bootstrap CI), Dice of matches, per-field count agreement."""
    per = []
    for (p, f), g in cells.groupby(["plate", "field"]):
        nr, nv = (g.source == "real").sum(), (g.source == "virtual").sum()
        tp = ((g.source == "real") & (g.match > 0)).sum()
        per.append({"plate": p, "field": f, "n_real": nr, "n_virtual": nv, "tp": tp})
    per = fields[["plate", "field", "well_id"]].merge(pd.DataFrame(per), how="left").fillna(0)

    def f1(tp, nr, nv):
        return 2 * tp.sum() / max(nr.sum() + nv.sum(), 1)

    tp, nr, nv = per.tp.sum(), per.n_real.sum(), per.n_virtual.sum()
    iou = cells.loc[(cells.source == "real") & (cells.match > 0), "iou"].to_numpy()
    return per, {"precision": float(tp / max(nv, 1)), "recall": float(tp / max(nr, 1)),
                 "f1": stats.bootstrap_ci(f1, per.tp, per.n_real, per.n_virtual, groups=per.well_id, n=1000),
                 "dice_matched_mean": float(np.mean(2 * iou / (1 + iou))) if len(iou) else float("nan"),
                 "n_real_cells": int(nr), "n_virtual_cells": int(nv), "n_matched": int(tp),
                 "mean_diam_real": float(cells[cells.source == "real"].groupby(["plate", "field"]).diam.first().mean()),
                 "mean_diam_virtual": float(cells[cells.source == "virtual"].groupby(["plate", "field"]).diam.first().mean()),
                 "count_agreement": stats.agreement_report(per.n_real, per.n_virtual, groups=per.well_id)}


def paired(cells):
    """Matched cells as aligned (real, virtual) frames."""
    r = cells[(cells.source == "real") & (cells.match > 0)]
    v = cells[cells.source == "virtual"].set_index(["plate", "field", "label"])
    v = v.loc[list(zip(r.plate, r.field, r.match))].reset_index()
    return r.reset_index(drop=True), v


def feature_ccc(r, v):
    out = {f: stats.lins_ccc(r[f], v[f]) for f in FEATURES}
    return {"per_feature": out, "median": float(np.median(list(out.values())))}


def phenotypes(cells, method):
    """Fit scaler, PCA(10) and states on REAL cells; assign every virtual cell with the same fitted model."""
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    real = cells.source == "real"
    sc = StandardScaler().fit(cells.loc[real, FEATURES])
    pca = PCA(N_PCA, random_state=SEED).fit(sc.transform(cells.loc[real, FEATURES]))
    Z = pca.transform(sc.transform(cells[FEATURES])).astype(np.float32)
    zr = Z[real.to_numpy()]
    if method == "kmeans":
        km = KMeans(K_MEANS, n_init=10, random_state=SEED).fit(zr)
        lab = km.predict(Z)
    else:
        import igraph as ig
        import leidenalg
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=15).fit(zr)
        _, ind = nn.kneighbors(zr)
        edges = np.column_stack([np.repeat(np.arange(len(zr)), ind.shape[1] - 1), ind[:, 1:].ravel()])
        g = ig.Graph(n=len(zr), edges=edges.tolist(), directed=False).simplify()
        part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition,
                                        resolution_parameter=LEIDEN_RES, seed=SEED)
        lr = np.array(part.membership)
        cent = np.stack([zr[lr == k].mean(0) for k in range(lr.max() + 1)])
        lab = np.empty(len(Z), int)
        lab[real.to_numpy()] = lr
        zv = Z[~real.to_numpy()]
        lab[~real.to_numpy()] = np.concatenate([((zv[i:i + 50000, None] - cent[None]) ** 2).sum(-1).argmin(1)
                                                for i in range(0, len(zv), 50000)])
    return lab, Z, {"pca_explained_var": float(pca.explained_variance_ratio_.sum())}


def state_agreement(cells):
    from sklearn.metrics import adjusted_rand_score
    r, v = paired(cells)
    ari = adjusted_rand_score(r.state, v.state)
    # Per-compound state proportions, real vs virtual (all cells of each source).
    prop = cells.groupby(["pert_iname", "source"]).state.value_counts(normalize=True).unstack(fill_value=0)
    ba = {}
    for s in prop.columns:
        pr = prop.xs("real", level="source")[s]
        pv = prop.xs("virtual", level="source")[s].reindex(pr.index, fill_value=0)
        if len(pr) >= 3:
            ba[str(s)] = {**stats.bland_altman(pr, pv), "mean_prop_real": float(pr.mean()),
                          "lins_ccc": stats.lins_ccc(pr, pv)}
    return {"ari": float(ari), "n_states": int(cells.loc[cells.source == "real", "state"].nunique()),
            "n_compounds": int(cells.pert_iname.nunique()), "state_proportions_bland_altman": ba}


# ------------------------------------------------------------------------------------------------ figures
def _rgb(h):
    return np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])


def _overlay(gray, mask, colour):
    lo, hi = np.percentile(gray, [1, 99.5])
    g = np.clip((gray - lo) / (hi - lo + 1e-6), 0, 1)
    img = np.repeat(g[..., None], 3, -1)
    img[find_boundaries(mask, mode="inner")] = _rgb(colour)
    return img


def fig_segmentation(examples, segs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [(f, s) for f in examples for s in segs]
    fig, ax = plt.subplots(len(rows), 3, figsize=(12, 4.1 * len(rows)), facecolor=BG)
    ax = np.atleast_2d(ax)
    for i, (row, seg) in enumerate(rows):
        key = f"{row.plate}_{row.field:04d}"
        arr, _ = D.load_plate(row.plate)
        lo, hi = D.norm_params(row.plate)
        x = D.normalise(np.asarray(arr[row.field]), lo, hi)
        v = np.load(CACHE / "virtual" / f"{key}.npy").astype(np.float32)
        mr = np.load(CACHE / "masks" / f"{key}_{seg}_real.npz")["mask"]
        mv = np.load(CACHE / "masks" / f"{key}_{seg}_virtual.npz")["mask"]
        bg = (lambda im: im[0]) if seg == "nuclei" else (lambda im: im[3])  # DNA or AGP under the outlines
        c = slice(CROP[0], CROP[0] + CROP[1])  # central crop so individual outlines are visible
        x, v, mr, mv = x[:, c, c], v[:, c, c], mr[c, c], mv[c, c]
        bf = x[C.CHANNELS.index("Brightfield")]
        lo_, hi_ = np.percentile(bf, [1, 99])
        ax[i, 0].imshow(np.clip((bf - lo_) / (hi_ - lo_), 0, 1), cmap="gray")
        ax[i, 1].imshow(_overlay(bg(x[C.TARGET_CH]), mr, REAL))
        ax[i, 2].imshow(_overlay(bg(v), mv, VIRT))
        under = "DNA" if seg == "nuclei" else "AGP"
        ax[i, 0].set_title(f"{row.plate} {row.well} s{row.site} ({row.pert_iname})\nbright-field, central {CROP[1]} px crop", fontsize=10)
        ax[i, 1].set_title(f"{seg}: real {under}, {len(np.unique(mr)) - 1} objects", fontsize=10, color="#222")
        ax[i, 2].set_title(f"{seg}: virtual {under}, {len(np.unique(mv)) - 1} objects", fontsize=10, color="#222")
        for a in ax[i]:
            a.set_xticks([]), a.set_yticks([])
            for sp in a.spines.values():
                sp.set_visible(False)
    from matplotlib.lines import Line2D
    fig.legend([Line2D([], [], color=REAL, lw=3), Line2D([], [], color=VIRT, lw=3)],
               ["real-stain masks", "virtual-stain masks"], loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)


def fig_umap(embeds, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(embeds), 2, figsize=(11, 5.2 * len(embeds)), facecolor=BG, squeeze=False)
    for i, (seg, (er, lr, ev, lv)) in enumerate(embeds.items()):
        # Fixed colour per state: the 7 largest real states get slots 1-7, the rest fold into grey "other".
        top = sorted(pd.Series(lr).value_counts().index[:7 if pd.Series(lr).nunique() > 8 else 8])
        cmap = {s: CAT[k] for k, s in enumerate(top)}
        for j, (e, l, name) in enumerate([(er, lr, "real"), (ev, lv, "virtual")]):
            a = ax[i, j]
            a.set_facecolor(BG)
            cols = [cmap.get(s, OTHER) for s in l]
            a.scatter(e[:, 0], e[:, 1], c=cols, s=1.5, lw=0, rasterized=True)
            a.set_title(f"{seg} ({SEGS[seg]['cluster']}): {name} cells, n={len(e):,}", fontsize=11)
            a.set_xticks([]), a.set_yticks([])
            for sp in a.spines.values():
                sp.set_color("#d0cfc8")
        from matplotlib.lines import Line2D
        h = [Line2D([], [], marker="o", ls="", color=cmap[s], ms=6) for s in top]
        lab = [f"state {s}" for s in top]
        if pd.Series(lr).nunique() > len(top):
            h.append(Line2D([], [], marker="o", ls="", color=OTHER, ms=6))
            lab.append("other states")
        ax[i, 1].legend(h, lab, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False, fontsize=9)
    fig.suptitle("States fitted on real cells; virtual cells assigned with the same model. UMAP fit on real, "
                 "virtual projected", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=BG)
    plt.close(fig)


def umap_embed(Z, lab, real):
    import umap
    rng = np.random.default_rng(SEED)
    ir, iv = np.where(real)[0], np.where(~real)[0]
    ir = rng.choice(ir, min(N_UMAP, len(ir)), replace=False)
    iv = rng.choice(iv, min(N_UMAP, len(iv)), replace=False)
    um = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEED).fit(Z[ir])
    return um.embedding_, lab[ir], um.transform(Z[iv]), lab[iv]


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["screen", "full"], required=True)
    ap.add_argument("--smoke", action="store_true", help="10 fields only")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    for name in ["cellpose", "cellpose.core", "cellpose.models", "cellpose.dynamics"]:
        logging.getLogger(name).setLevel(logging.WARNING)
    t_start = time.time()
    out = OUT / (a.stage + ("_smoke" if a.smoke else ""))
    for d in [out, CACHE / "virtual", CACHE / "masks", CACHE / "cells"]:
        d.mkdir(parents=True, exist_ok=True)

    if a.stage == "screen":
        segs = list(SEGS)
    else:
        src = OUT / ("screen_smoke" if a.smoke else "screen") / "report.json"
        segs = json.loads(src.read_text())["selected_for_full"]
    fields = select_fields(all_fields(), a.stage, a.smoke)
    log.info("stage %s: %d fields, segmentations %s", a.stage, len(fields), segs)

    eng, timing, tabs = Engines(), {"predict": 0.0, "calibrate": 0.0, "segment": 0.0}, []
    cal = calibrate_diameters(all_fields(), segs, eng, timing)
    t_loop = time.time()
    for k, row in enumerate(fields.itertuples()):
        tabs += process_field(row, segs, eng, timing, cal)
        if (k + 1) % 5 == 0 or k + 1 == len(fields):
            el = time.time() - t_loop
            log.info("%d/%d fields  %.1fs/field  (predict %.0fs, segment %.0fs, device %s)", k + 1, len(fields),
                     el / (k + 1), timing["predict"], timing["segment"], eng.dev)
    t_fields = time.time() - t_loop
    cells = pd.concat(tabs, ignore_index=True)

    report = {"stage": a.stage, "smoke": a.smoke, "n_fields": len(fields),
              "n_fields_dmso": int((fields.pert_iname == "DMSO").sum()),
              "n_fields_heldout_compound": int(((fields.split == "test")).sum()),
              "device": eng.dev, "iou_threshold": IOU_MIN, "cellpose_diameters": cal, "features": FEATURES, "paths": {}}
    rows, embeds, field_tabs = [], {}, []
    for seg in segs:
        c = cells[cells.seg == seg].reset_index(drop=True)
        per, sm = seg_metrics(c, fields)
        per["seg"] = seg
        field_tabs.append(per)
        r, v = paired(c)
        fc = feature_ccc(r, v)
        lab, Z, pinfo = phenotypes(c, SEGS[seg]["cluster"])
        c["state"] = lab
        cells.loc[cells.seg == seg, "state"] = lab
        ph = {**state_agreement(c), **pinfo}
        embeds[seg] = umap_embed(Z, lab, (c.source == "real").to_numpy())
        report["paths"][seg] = {"virtual_path": "virtual_stains -> " + SEGS[seg]["label"] + " -> f1_vs_real",
                                "reference": "real_stains -> " + SEGS[seg]["label"],
                                "segmentation": sm, "features_ccc": fc, "phenotypes": ph}
        ca = sm["count_agreement"]
        rows.append({"path": "virtual_stains -> " + SEGS[seg]["label"], "seg": seg,
                     "precision": sm["precision"], "recall": sm["recall"], "f1": sm["f1"]["estimate"],
                     "f1_ci_low": sm["f1"]["ci_low"], "f1_ci_high": sm["f1"]["ci_high"],
                     "dice_matched": sm["dice_matched_mean"],
                     "count_ccc": ca["lins_ccc"]["estimate"], "count_ccc_ci_low": ca["lins_ccc"]["ci_low"],
                     "count_ccc_ci_high": ca["lins_ccc"]["ci_high"], "count_bias": ca["bland_altman"]["bias"],
                     "median_feature_ccc": fc["median"], "ari": ph["ari"], "n_states": ph["n_states"],
                     "n_real_cells": sm["n_real_cells"], "n_virtual_cells": sm["n_virtual_cells"],
                     "n_matched": sm["n_matched"]})
    paths = pd.DataFrame(rows)
    paths.to_csv(out / "paths.csv", index=False)
    pd.concat(field_tabs).to_csv(out / "fields.csv", index=False)
    cells["state"] = cells["state"].astype(int)
    cells.to_parquet(out / "cells.parquet", index=False)

    if a.stage == "screen":
        best = paths.f1.max()
        report["selected_for_full"] = paths.loc[paths.f1 >= best - F1_TIE, "seg"].tolist()
        report["selection_rule"] = f"virtual-vs-real F1 within {F1_TIE} of the best"

    # Example figure: one DMSO and one compound field, closest to the median F1 of the first segmentation.
    per = field_tabs[0]
    per = per.assign(f1=2 * per.tp / (per.n_real + per.n_virtual).clip(lower=1)).merge(fields)
    ex = []
    for is_dmso in [True, False]:
        g = per[(per.pert_iname == "DMSO") == is_dmso]
        if len(g):
            ex.append(g.iloc[(g.f1 - per.f1.median()).abs().argsort().iloc[0]])
    fig_segmentation([fields[(fields.plate == e.plate) & (fields.field == e.field)].iloc[0] for e in ex],
                     segs, out / "fig_q3_segmentation_examples.png")
    fig_umap(embeds, out / "fig_q3_umap.png")
    report["runtime_s"] = {"fields": round(t_fields, 1), "total": round(time.time() - t_start, 1),
                           "per_field": round(t_fields / len(fields), 2), **{k: round(v, 1) for k, v in timing.items()}}
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))
    print(paths.round(3).to_string(index=False))
    print("selected_for_full:", report.get("selected_for_full"), "| runtime", report["runtime_s"])


if __name__ == "__main__":
    main()
