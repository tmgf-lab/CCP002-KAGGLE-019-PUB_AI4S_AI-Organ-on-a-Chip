"""Q3 guided search (sweep/configs/q3_single_cell.yaml): segmentation search, then downstream search on the winner.

  predict    virtual stains for all 1,536 test fields (cached, float16)                       [GPU job]
  segsearch  engine Search over channels x preprocessing x segmenter x diameter x thresholds;
             score = F1 (IoU >= 0.5) of virtual vs real masks, both made with the same path    [GPU job]
  features   per-cell tables (shape, intensity, Haralick, Zernike) for the best segmentation      [GPU job]
  downstream engine Search over features x processing x embedding x clustering x n_states;
             states fitted on REAL cells, virtual cells assigned by the same model; score = ARI  [CPU job]
  supervised engine Search: per-cell treated-vs-DMSO classifier on virtual features (AUC)          [CPU job]
  assess     masks F1/Dice/AP(0.5-0.95), counts agreement, feature CCC, ARI/NMI, bootstrap stability,
             per-compound Wasserstein/KS, mixed model for state proportions                       [CPU job]

All data subsets are nested 25/50/100% of the 1,536 fields (engine.nested_subset on field ids).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
import data as D  # noqa: E402
from sweep import engine, stats  # noqa: E402
from sweep.q3_single_cell import STAINS, all_fields, match  # noqa: E402

OUT = Path(os.environ.get("Q3_OUT", C.RUNS / "q3"))
VCACHE = Path(os.environ.get("Q3_CACHE", OUT / "cache")) / "virtual"
MCACHE = Path(os.environ.get("Q3_CACHE", OUT / "cache")) / "masks"
SEED = 0
SEG_TREE = {"channels": ["dna_only", "dna+agp", "all_five"], "preprocessing": ["none", "background_subtraction"],
            "seg_model": ["cellpose_nuclei", "cellpose_cyto3", "cellpose_cyto2", "stardist", "otsu_watershed"],
            "diameter": ["auto", "10", "14", "18", "30"], "flow_threshold": ["0.4", "0.6"], "cellprob_threshold": ["-1", "0"]}
SEG_BASE = {"channels": "dna+agp", "preprocessing": "none", "seg_model": "cellpose_cyto3", "diameter": "30",
            "flow_threshold": "0.4", "cellprob_threshold": "-1"}
DOWN_TREE = {"features": ["morphology", "morphology+intensity", "morphology+intensity+haralick", "+zernike"],
             "processing": ["standardise", "robust_z_vs_dmso", "correlation_filter_0.9", "pca"],
             "embedding": ["pca", "umap"],
             "clustering": ["kmeans_4_16", "gmm_bic", "leiden", "louvain", "hdbscan", "hierarchical_ward"],
             "n_states_choice": ["silhouette", "bic", "bootstrap_stability"]}
DOWN_BASE = {"features": "morphology+intensity", "processing": "standardise", "embedding": "pca",
             "clustering": "kmeans_4_16", "n_states_choice": "silhouette"}
SUP_TREE = {"features": DOWN_TREE["features"], "processing": ["standardise", "robust_z_vs_dmso", "pca"],
            "model": ["logistic", "gradient_boosting", "mlp"]}
SUP_BASE = {"features": "morphology+intensity", "processing": "standardise", "model": "logistic"}
SHAPE = ["area", "eccentricity", "solidity", "perimeter", "major_axis_length", "minor_axis_length", "extent"]
HARALICK = ["contrast", "homogeneity", "energy", "correlation"]


def _has(mod):
    try:
        __import__(mod)
        return True
    except Exception:  # noqa: BLE001
        return False


HAS_STARDIST, HAS_MAHOTAS = _has("stardist"), _has("mahotas")


class SegRules:
    def __call__(self, p):
        m = p["seg_model"]
        if m in ("cellpose_nuclei", "stardist", "otsu_watershed") and p["channels"] != "dna_only":
            return False
        if m in ("cellpose_cyto3", "cellpose_cyto2") and p["channels"] == "dna_only":
            return False
        if m == "stardist" and not HAS_STARDIST:
            return False
        if not m.startswith("cellpose") and (p["diameter"], p["flow_threshold"], p["cellprob_threshold"]) != ("auto", "0.4", "-1"):
            return False  # parameters only exist for Cellpose: one canonical value avoids duplicate paths
        return True


class DownRules:
    def __call__(self, p):
        if p["features"] == "+zernike" and not HAS_MAHOTAS:
            return False
        if p["clustering"] in ("leiden", "louvain", "hdbscan") and p["n_states_choice"] != "silhouette":
            return False  # these pick their own number of states: one canonical value
        if p["clustering"] == "gmm_bic" and p["n_states_choice"] != "bic":
            return False
        return True


class SupRules:
    def __call__(self, p):
        return not (p["features"] == "+zernike" and not HAS_MAHOTAS)


# ------------------------------------------------------------------------------------------ fields and images
_F = {}


def fields():
    if "f" not in _F:
        _F["f"] = all_fields()
    return _F["f"]


def field_images(i):
    f = fields().iloc[i]
    arr, _ = D.load_plate(f.plate)
    lo, hi = D.norm_params(f.plate)
    x = D.normalise(np.asarray(arr[f.field]), lo, hi)
    v = np.load(VCACHE / f"{f.plate}_{f.field:04d}.npy").astype(np.float32)
    return x[C.TARGET_CH], v


def cmd_predict(a):
    VCACHE.mkdir(parents=True, exist_ok=True)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if not a.winner:  # default: the Q1 winner from the GitHub release (weights.py)
        from weights import winner_json
        a.winner = str(winner_json())
    w = json.loads(Path(a.winner).read_text())  # the Q1 winner: its seeds and its inference mode
    ms = [D.load_model(Path(d), dev=dev) for d in w["run_dirs"]]
    cfg = ms[0][1]
    models = [m for m, _ in ms] if w["inference"].startswith("seed_ensemble") else [ms[0][0]]

    def infer(x):
        return D.predict_multi(models, cfg, x, dev, w["inference"].endswith("tta"))
    f = fields()
    todo = [i for i in range(len(f)) if not (VCACHE / f"{f.plate.iloc[i]}_{f.field.iloc[i]:04d}.npy").exists()]
    if a.smoke:
        todo = todo[:24]
    t0 = time.time()
    for k, i in enumerate(todo):
        r = f.iloc[i]
        arr, _ = D.load_plate(r.plate)
        lo, hi = D.norm_params(r.plate)
        np.save(VCACHE / f"{r.plate}_{r.field:04d}.npy",
                infer(D.normalise(np.asarray(arr[r.field]), lo, hi)).astype(np.float16))
        if k % 200 == 0:
            print(f"  predict {k}/{len(todo)} ({time.time() - t0:.0f}s, {dev})", flush=True)
    print(f"predicted {len(todo)} fields in {time.time() - t0:.0f}s on {dev}")


# ------------------------------------------------------------------------------------------ segmentation
_SEG = {}


def _device():
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    return f"cuda:{int(os.environ.get('SEARCH_WORKER', 0)) % torch.cuda.device_count()}"


def seg_inputs(stains, p):
    dna = stains[0]
    cyto = {"dna_only": None, "dna+agp": stains[3], "all_five": stains[1:].mean(0)}[p["channels"]]
    if p["preprocessing"] == "background_subtraction":
        from scipy.ndimage import gaussian_filter
        dna = np.clip(dna - gaussian_filter(dna, 25), 0, None)
        cyto = None if cyto is None else np.clip(cyto - gaussian_filter(cyto, 25), 0, None)
    return dna, cyto


def segment(stains, p):
    m = p["seg_model"]
    dna, cyto = seg_inputs(stains, p)
    if m == "otsu_watershed":
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max
        from skimage.filters import threshold_otsu
        from skimage.segmentation import watershed
        sm = ndi.gaussian_filter(dna, 2)
        fg = sm > threshold_otsu(sm)
        dist = ndi.distance_transform_edt(fg)
        pk = peak_local_max(dist, min_distance=5, labels=fg, exclude_border=False)
        mk = np.zeros_like(dna, int)
        mk[tuple(pk.T)] = np.arange(1, len(pk) + 1)
        return watershed(-dist, mk, mask=fg).astype(np.uint16)
    if m == "stardist":
        if "sd" not in _SEG:
            from stardist.models import StarDist2D
            _SEG["sd"] = StarDist2D.from_pretrained("2D_versatile_fluo")
        from csbdeep.utils import normalize
        return _SEG["sd"].predict_instances(normalize(dna, 1, 99.8))[0].astype(np.uint16)
    from cellpose import models
    dev = _device()
    key = (m, dev)
    if key not in _SEG:
        import torch
        _SEG[key] = models.CellposeModel(gpu=dev != "cpu", device=torch.device(dev), model_type=m.replace("cellpose_", ""))
    diam = None if p["diameter"] == "auto" else float(p["diameter"])
    kw = dict(diameter=diam if diam else (17.0 if m == "cellpose_nuclei" else 30.0),
              flow_threshold=float(p["flow_threshold"]), cellprob_threshold=float(p["cellprob_threshold"]))
    if cyto is None:
        mask = _SEG[key].eval(dna, channels=[0, 0], **kw)[0]
    else:
        mask = _SEG[key].eval(np.stack([cyto, dna], -1), channels=[1, 2], channel_axis=-1, **kw)[0]
    return mask.astype(np.uint16)


def seg_key(p):
    return "_".join(p[k] for k in SEG_TREE).replace(".", "p").replace("+", "")


def masks_for(i, p):
    d = MCACHE / seg_key(p)
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{i:04d}.npz"
    if f.exists():
        z = np.load(f)
        return z["real"], z["virtual"]
    real, virt = field_images(i)
    mr, mv = segment(real, p), segment(virt, p)
    np.savez_compressed(f, real=mr, virtual=mv)
    return mr, mv


def seg_score(p, frac, trial=None):
    n = 24 if os.environ.get("Q3_SMOKE") == "1" else len(fields())
    ids = engine.nested_subset(n, frac, SEED)
    tp = nr = nv = 0
    for i in ids:
        mr, mv = masks_for(int(i), p)
        pairs, _ = match(mr, mv)
        tp, nr, nv = tp + len(pairs), nr + int(mr.max()), nv + int(mv.max())
    return 2 * tp / max(nr + nv, 1)


# ------------------------------------------------------------------------------------------ per-cell features
def cell_features(mask, stains, with_haralick=True, with_zernike=True):
    from skimage.measure import regionprops
    rows = []
    for rp in regionprops(mask.astype(np.int32)):
        r = {"label": rp.label, "y": rp.centroid[0], "x": rp.centroid[1]}
        for k in SHAPE:
            r[k] = float(getattr(rp, k))
        sl, m = rp.slice, rp.image
        for c, s in enumerate(STAINS):
            px = stains[c][sl][m]
            r[f"mean_{s}"], r[f"std_{s}"] = float(px.mean()), float(px.std())
            r[f"integ_{s}"] = float(px.sum())
        if with_haralick:
            from skimage.feature import graycomatrix, graycoprops
            for c, s in ((0, "DNA"), (3, "AGP")):
                crop = stains[c][sl] * m
                q = np.clip(crop / (np.percentile(stains[c], 99.5) + 1e-6) * 31, 0, 31).astype(np.uint8)
                gm = graycomatrix(q, [1], [0, np.pi / 2], levels=32, symmetric=True, normed=True)
                for h in HARALICK:
                    r[f"har_{h}_{s}"] = float(graycoprops(gm, h).mean())
        if with_zernike and HAS_MAHOTAS:
            import mahotas
            z = mahotas.features.zernike_moments(m.astype(np.uint8), max(m.shape) / 2, degree=4)
            for j, v in enumerate(z[:6]):
                r[f"zern_{j}"] = float(v)
        rows.append(r)
    if not rows:  # a field with no cells (e.g. all cells killed): empty table with the usual columns
        return pd.DataFrame(columns=["label", "y", "x", *SHAPE])
    return pd.DataFrame(rows)


def cmd_features(a):
    given = a.seg_path or os.environ.get("Q3_SEG_PATH")
    best = json.loads(given) if given else best_from(OUT / "seg", "q3_seg", SEG_TREE)
    print("segmentation path:", best)
    f = fields()
    ids = range(len(f)) if not a.smoke else range(24)
    tabs, per = [], []
    t0 = time.time()
    for k, i in enumerate(ids):
        real, virt = field_images(i)
        mr, mv = masks_for(i, best)
        pairs, iou = match(mr, mv)
        for src, mask, img, col in (("real", mr, real, 0), ("virtual", mv, virt, 1)):
            t = cell_features(mask, img)
            t["match"] = t.label.map(dict(zip(pairs[:, col], pairs[:, 1 - col]))).fillna(0).astype(int)
            t["iou"] = t.label.map(dict(zip(pairs[:, col], iou)))
            t["source"], t["field_id"] = src, i
            tabs.append(t)
        per.append({"field_id": i, "n_real": int(mr.max()), "n_virtual": int(mv.max()), "tp": len(pairs),
                    "iou": iou.tolist()})
        if k % 200 == 0:
            print(f"  features {k}/{len(ids)} ({time.time() - t0:.0f}s)", flush=True)
    cells = pd.concat(tabs, ignore_index=True)
    meta = f.reset_index(drop=True)[["plate", "field", "well", "site", "well_id", "pert_iname", "split"]]
    cells = cells.merge(meta, left_on="field_id", right_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cells.to_parquet(OUT / "cells.parquet", index=False)
    json.dump({"seg_path": best, "per_field": per}, open(OUT / "seg_fields.json", "w"))
    print(f"{len(cells)} cells -> {OUT / 'cells.parquet'}")


# ------------------------------------------------------------------------------------------ downstream
_CELLS = {}


def cells_table():
    if "c" not in _CELLS:
        c = pd.read_parquet(Path(os.environ.get("Q3_CELLS", OUT / "cells.parquet")))
        _CELLS["c"] = c
    return _CELLS["c"]


def feature_cols(c, fs):
    base = SHAPE[:]
    if fs != "morphology":
        base += [x for x in c.columns if x.startswith(("mean_", "std_", "integ_"))]
    if "haralick" in fs or fs == "+zernike":
        base += [x for x in c.columns if x.startswith("har_")]
    if fs == "+zernike":
        base += [x for x in c.columns if x.startswith("zern_")]
    return [x for x in base if x in c.columns]


def processed(c, p, cols, real_mask, rng):
    """Scaling fitted on REAL cells (robust z: on real DMSO cells); returns Z for all rows and the transform info."""
    X = c[cols].to_numpy(float)
    X = np.nan_to_num(X)
    if p["processing"] == "robust_z_vs_dmso":
        ref = X[real_mask & (c.pert_iname == "DMSO").to_numpy()]
        med = np.median(ref, 0)
        mad = 1.4826 * np.median(np.abs(ref - med), 0) + 1e-9
        Z = (X - med) / mad
    else:
        mu, sd = X[real_mask].mean(0), X[real_mask].std(0) + 1e-9
        Z = (X - mu) / sd
    if p["processing"] == "correlation_filter_0.9":
        R = np.abs(np.corrcoef(Z[real_mask][rng.choice(real_mask.sum(), min(20000, real_mask.sum()), replace=False)].T))
        keep = []
        for j in range(Z.shape[1]):
            if not keep or np.nanmax(R[j, keep]) <= 0.9:
                keep.append(j)
        Z = Z[:, keep]
    Z = np.clip(Z, -20, 20)
    if p["processing"] == "pca" or p["embedding"] in ("pca", "umap"):
        from sklearn.decomposition import PCA
        pca = PCA(min(10, Z.shape[1]), random_state=SEED).fit(Z[real_mask])
        Z = pca.transform(Z)
    if p["embedding"] == "umap":
        import umap
        sub = np.where(real_mask)[0]
        sub = rng.choice(sub, min(20000, len(sub)), replace=False)
        um = umap.UMAP(n_components=5, random_state=SEED, n_jobs=1).fit(Z[sub])
        Z = np.vstack([um.transform(Z[i:i + 50000]) for i in range(0, len(Z), 50000)])
    return Z.astype(np.float32)


def fit_states(Zr, p, rng):
    """Fit states on (a subsample of) real cells -> centroids; returns (labels for Zr subsample fit, centroids)."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    n_fit = min(len(Zr), 30000 if p["clustering"] != "hierarchical_ward" else 15000)
    S = Zr[rng.choice(len(Zr), n_fit, replace=False)]
    cl = p["clustering"]
    if cl in ("kmeans_4_16", "hierarchical_ward", "gmm_bic"):
        ks = range(4, 17, 2)
        best_k, best_s = 6, -np.inf
        for k in ks:
            if cl == "gmm_bic" or p["n_states_choice"] == "bic":
                from sklearn.mixture import GaussianMixture
                g = GaussianMixture(k, covariance_type="diag", random_state=SEED).fit(S[:10000])
                s = -g.bic(S[:10000])
            elif p["n_states_choice"] == "bootstrap_stability":
                from sklearn.metrics import adjusted_rand_score
                a_ = KMeans(k, n_init=3, random_state=1).fit(S[:8000])
                b_ = KMeans(k, n_init=3, random_state=2).fit(S[rng.choice(len(S), 8000, replace=True)])
                s = adjusted_rand_score(a_.predict(S[:8000]), b_.predict(S[:8000]))
            else:
                lab = KMeans(k, n_init=3, random_state=SEED).fit_predict(S[:8000])
                s = silhouette_score(S[:8000], lab, sample_size=4000, random_state=SEED)
            if s > best_s:
                best_k, best_s = k, s
        if cl == "kmeans_4_16":
            lab = KMeans(best_k, n_init=5, random_state=SEED).fit_predict(S)
        elif cl == "gmm_bic":
            from sklearn.mixture import GaussianMixture
            lab = GaussianMixture(best_k, covariance_type="diag", random_state=SEED).fit_predict(S)
        else:
            from sklearn.cluster import AgglomerativeClustering
            lab = AgglomerativeClustering(best_k, linkage="ward").fit_predict(S)
    elif cl in ("leiden", "louvain"):
        import igraph as ig
        from sklearn.neighbors import NearestNeighbors
        _, ind = NearestNeighbors(n_neighbors=15).fit(S).kneighbors(S)
        g = ig.Graph(n=len(S), edges=np.column_stack([np.repeat(np.arange(len(S)), 14), ind[:, 1:].ravel()]).tolist()).simplify()
        if cl == "leiden":
            import leidenalg
            lab = np.array(leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, resolution_parameter=1.0,
                                                    seed=SEED).membership)
        else:
            lab = np.array(g.community_multilevel().membership)
    else:  # hdbscan
        from sklearn.cluster import HDBSCAN
        lab = HDBSCAN(min_cluster_size=max(50, len(S) // 200)).fit_predict(S[:20000])
        S = S[:20000]
        if (lab >= 0).sum() == 0:
            lab = np.zeros(len(S), int)
        S, lab = S[lab >= 0], lab[lab >= 0]
    cent = np.stack([S[lab == k].mean(0) for k in np.unique(lab)])
    return cent


def assign(Z, cent):
    return np.concatenate([((Z[i:i + 50000, None] - cent[None]) ** 2).sum(-1).argmin(1) for i in range(0, len(Z), 50000)])


def down_states(p, frac, rng=None):
    rng = rng or np.random.default_rng(SEED)
    c = cells_table()
    fids = np.sort(c.field_id.unique())
    c = c[c.field_id.isin(fids[engine.nested_subset(fids, frac, SEED)])].reset_index(drop=True)
    cols = feature_cols(c, p["features"])
    real = (c.source == "real").to_numpy()
    Z = processed(c, p, cols, real, rng)
    cent = fit_states(Z[real], p, rng)
    return c, Z, cent, assign(Z, cent)


def paired_idx(c):
    r = c[(c.source == "real") & (c.match > 0)]
    v = c[c.source == "virtual"].reset_index().set_index(["field_id", "label"])["index"]
    vi = v.reindex(list(zip(r.field_id, r.match))).to_numpy()
    ok = ~np.isnan(vi)
    return r.index.to_numpy()[ok], vi[ok].astype(int)


def down_score(p, frac, trial=None):
    from sklearn.metrics import adjusted_rand_score
    c, Z, cent, lab = down_states(p, frac)
    ri, vi = paired_idx(c)
    return float(adjusted_rand_score(lab[ri], lab[vi])) if len(cent) > 1 else 0.0


def sup_score(p, frac, trial=None):
    """Per-cell treated vs DMSO on VIRTUAL cells, CV grouped by well (subsample 40k cells for speed)."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    rng = np.random.default_rng(SEED)
    c = cells_table()
    fids = np.sort(c.field_id.unique())
    src = os.environ.get("Q3_SUP_SOURCE", "virtual")
    c = c[c.field_id.isin(fids[engine.nested_subset(fids, frac, SEED)]) & (c.source == src)].reset_index(drop=True)
    if len(c) > 40000:
        c = c.iloc[np.sort(rng.choice(len(c), 40000, replace=False))].reset_index(drop=True)
    cols = feature_cols(c, p["features"])
    real = np.ones(len(c), bool)
    Z = processed(c, {**p, "embedding": "none"}, cols, real, rng)
    y = (c.pert_iname != "DMSO").astype(int).to_numpy()
    g = (c.plate + "_" + c.well).to_numpy()
    if len(np.unique(y)) < 2:
        return 0.5
    oof = np.full(len(y), 0.5)
    for tr, te in GroupKFold(5).split(Z, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        oof[te] = make_sup(p["model"]).fit(Z[tr], y[tr]).predict_proba(Z[te])[:, 1]
    return float(roc_auc_score(y, oof))


def make_sup(name):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    return {"logistic": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "gradient_boosting": HistGradientBoostingClassifier(class_weight="balanced", max_iter=150, random_state=SEED),
            "mlp": MLPClassifier((64,), early_stopping=True, max_iter=200, random_state=SEED)}[name]


# ------------------------------------------------------------------------------------------ search plumbing
def best_from(d, name, tree):
    df = pd.read_csv(Path(d) / f"{name}_paths.csv")
    df = df[(df.state == "COMPLETE") & df.skipped.isna()].sort_values("value", ascending=False)
    return {k: str(df.iloc[0][k]) for k in tree}


def run(name, tree, base, rules, fn, budget, out, a):
    s = engine.Search(name, tree, out, budget=budget, rules=rules, baseline=base, warmup_per_option=1 if a.smoke else 3)
    summ = s.run(fn, workers=a.workers, hours=a.hours)
    print(f"{name}: evaluated {summ['evaluated']}, completed {summ['completed_100pct']}, excluded "
          f"{[(e['node'], e['option']) for e in summ['excluded_options']]}, best {summ['best'][0]['value'] if summ['best'] else None}")


# ------------------------------------------------------------------------------------------ assessment
def cmd_assess(a):
    from scipy.stats import ks_2samp, wasserstein_distance
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    out = OUT
    seg = json.load(open(out / "seg_fields.json"))
    per = pd.DataFrame(seg["per_field"])
    f = fields().reset_index(drop=True)
    per["well_id"] = f.loc[per.field_id, "well_id"].to_numpy()
    ious = np.concatenate([np.array(x) for x in per.iou if len(x)]) if per.iou.map(len).sum() else np.zeros(0)
    nr, nv = per.n_real.sum(), per.n_virtual.sum()
    ap = {f"{t:.2f}": float((ious >= t).sum() / max(nr + nv - (ious >= t).sum(), 1)) for t in np.arange(0.5, 0.96, 0.05)}
    rep = {"seg_path": seg["seg_path"],
           "f1": stats.bootstrap_ci(lambda tp, a_, b_: 2 * tp.sum() / max(a_.sum() + b_.sum(), 1), per.tp, per.n_real,
                                    per.n_virtual, groups=per.well_id, n=1000),
           "dice_matched": float(np.mean(2 * ious / (1 + ious))) if len(ious) else None,
           "ap_by_iou": ap, "ap_mean_0.5_0.95": float(np.mean(list(ap.values()))),
           "count_agreement": stats.agreement_report(per.n_real, per.n_virtual, groups=per.well_id)}
    c = cells_table()
    ri, vi = paired_idx(c)
    cols = feature_cols(c, "+zernike" if HAS_MAHOTAS else "morphology+intensity+haralick")
    rep["feature_ccc"] = {k: stats.lins_ccc(c.loc[ri, k], c.loc[vi, k]) for k in cols}
    rep["feature_ccc_median"] = float(np.median(list(rep["feature_ccc"].values())))
    # per-compound distribution distances (median over compounds) for 6 key features
    dist = {}
    for k in ["area", "mean_DNA", "mean_ER", "mean_RNA", "mean_AGP", "mean_Mito"]:
        w, ks = [], []
        for comp, g in c.groupby("pert_iname"):
            r_, v_ = g.loc[g.source == "real", k], g.loc[g.source == "virtual", k]
            if len(r_) > 20 and len(v_) > 20:
                w.append(wasserstein_distance(r_, v_) / (r_.std() + 1e-9))
                ks.append(ks_2samp(r_, v_).statistic)
        dist[k] = {"wasserstein_over_sd_median": float(np.median(w)), "ks_median": float(np.median(ks)), "n_compounds": len(w)}
    rep["per_compound_distance"] = dist
    # best downstream path: ARI, NMI, bootstrap stability, mixed model on state proportions
    pf = Path(a.paths_from) if a.paths_from else out  # replication study: winning paths of the A549 study
    if (pf / "down" / "q3_down_paths.csv").exists():
        bp = best_from(pf / "down", "q3_down", DOWN_TREE)
        cc, Z, cent, lab = down_states(bp, 1.0)
        ri2, vi2 = paired_idx(cc)
        stab = []
        real = (cc.source == "real").to_numpy()
        for b in range(5 if not a.smoke else 2):
            cent_b = fit_states(Z[real], bp, np.random.default_rng(100 + b))
            stab.append(adjusted_rand_score(lab[real], assign(Z[real], cent_b)))
        cc["state"] = lab
        prop = cc.groupby(["plate", "well", "source"]).state.value_counts(normalize=True).rename("prop").reset_index()
        import statsmodels.formula.api as smf
        mm = {}
        for st_ in sorted(prop.state.unique())[:16]:
            dd = prop[prop.state == st_].pivot_table(index=["plate", "well"], columns="source", values="prop", fill_value=0)
            dd = dd.reset_index().melt(id_vars=["plate", "well"], value_name="prop", var_name="source")
            try:
                fit = (smf.mixedlm("prop ~ source", dd, groups=dd["plate"]).fit(disp=False) if dd.plate.nunique() > 1
                       else smf.ols("prop ~ source", dd).fit())  # one plate: no plate random effect
                mm[str(st_)] = {"virtual_minus_real": float(fit.params.get("source[T.virtual]", np.nan)),
                                "p": float(fit.pvalues.get("source[T.virtual]", np.nan))}
            except Exception as e:  # noqa: BLE001
                mm[str(st_)] = f"failed: {e}"
        rep["states"] = {"path": bp, "n_states": int(len(cent)), "ari": float(adjusted_rand_score(lab[ri2], lab[vi2])),
                         "nmi": float(normalized_mutual_info_score(lab[ri2], lab[vi2])),
                         "bootstrap_stability_ari": [float(x) for x in stab], "mixed_model_state_proportions": mm}
    if a.paths_from and (pf / "sup_virtual" / "q3_sup_virtual_paths.csv").exists():
        rep["supervised_best"], rep["supervised_auc"] = {}, {}
        for s in ("virtual", "real"):
            os.environ["Q3_SUP_SOURCE"] = s
            rep["supervised_best"][s] = best_from(pf / f"sup_{s}", f"q3_sup_{s}", SUP_TREE)
            rep["supervised_auc"][s] = sup_score(rep["supervised_best"][s], 1.0)
    elif (out / "sup_virtual" / "q3_sup_virtual_paths.csv").exists():
        rep["supervised_best"] = {s: best_from(out / f"sup_{s}", f"q3_sup_{s}", SUP_TREE) for s in ("virtual", "real")
                                  if (out / f"sup_{s}" / f"q3_sup_{s}_paths.csv").exists()}
        rep["supervised_auc"] = {s: float(pd.read_csv(out / f"sup_{s}" / f"q3_sup_{s}_paths.csv").query("state=='COMPLETE'").value.max())
                                 for s in rep["supervised_best"]}
    json.dump(rep, open(out / "q3_report.json", "w"), indent=1, default=str)
    print(json.dumps({k: rep[k] for k in ("seg_path", "f1", "ap_mean_0.5_0.95", "feature_ccc_median")}, indent=1, default=str))
    if "states" in rep:
        print("states:", {k: rep["states"][k] for k in ("n_states", "ari", "nmi", "bootstrap_stability_ari")})
    if "supervised_auc" in rep:
        print("supervised AUC:", rep["supervised_auc"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["predict", "segsearch", "features", "downstream", "supervised", "assess"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--budget", type=int)
    ap.add_argument("--seg-path", help="JSON path dict to use instead of the search winner")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--winner", help="predict: q1_winner.json (default: the Q1 winner, downloaded by weights.py)")
    ap.add_argument("--paths-from", help="assess: winning downstream/supervised paths of this Q3 folder (replication)")
    a = ap.parse_args()
    if a.smoke:
        os.environ["Q3_SMOKE"] = "1"
    OUT.mkdir(parents=True, exist_ok=True)
    if a.command == "predict":
        cmd_predict(a)
    elif a.command == "segsearch":
        run("q3_seg", SEG_TREE, SEG_BASE, SegRules(), seg_score, a.budget or (6 if a.smoke else 50), OUT / "seg", a)
    elif a.command == "features":
        cmd_features(a)
    elif a.command == "downstream":
        run("q3_down", DOWN_TREE, DOWN_BASE, DownRules(), down_score, a.budget or (8 if a.smoke else 200), OUT / "down", a)
    elif a.command == "supervised":
        for src in ("virtual", "real"):
            os.environ["Q3_SUP_SOURCE"] = src
            run(f"q3_sup_{src}", SUP_TREE, SUP_BASE, SupRules(), sup_score, a.budget or (6 if a.smoke else 30),
                OUT / f"sup_{src}", a)
    else:
        cmd_assess(a)


if __name__ == "__main__":
    main()
