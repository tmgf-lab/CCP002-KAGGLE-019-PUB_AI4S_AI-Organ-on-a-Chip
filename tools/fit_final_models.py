"""Fit the final Q2, Q3 and Q4 models for the end-to-end system (vcp_system.py), using exactly the winning paths of the
searches (nothing re-chosen), and score each one on data it never saw.

  Q2  the 4 winning pipelines on virtual-stain features, fitted on plate BR00117017 (the "calibration plate");
      held-out check on plate BR00117019 (wells never used for fitting).
  Q3  the winning cell-state path (morphology -> robust z vs DMSO -> PCA -> Ward), fitted on the real cells of BR00117017;
      held-out check on BR00117019: real vs virtual cells assigned to the same states (ARI).
  Q4  the winning chip-QC path (DINOv2-S + hand-crafted, correlation filter, L1 logistic), fitted on all chip
      experiments except one held-out experiment per cell type; held-out check on those 6 experiments.

The cross-validated scores in the report stay the reported ones; these held-out numbers check the final fitted models.

    python tools/fit_final_models.py [--raw <raw runs folder>] [--chips <OOC image folder>]
Writes models/q2_final.pkl, models/q3_states.json, models/q4_chip_qc.json and models/final_models.json.
"""
import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "models"
CAL, HELD = "BR00117017", "BR00117019"
LABELS = ["toxic_0.5", "toxic_0.8", "viability", "treated_vs_dmso"]


def fit_q2():
    """The 4 winning Q2 pipelines (virtual-stain features) fitted on the calibration plate. Deterministic (seed 0); needs
    only results/ (the repository), so vcp_system.py can redo it when the saved pickle does not fit the installed sklearn."""
    from sweep import q2_models as M
    from sweep import q2_search as S
    sites = pd.read_csv(ROOT / "results" / "a549" / "q1_test" / "site_features.csv")
    tmp = Path(tempfile.mkdtemp())
    sites[sites.plate == CAL].to_csv(tmp / "cal.csv", index=False)
    paths = json.loads((ROOT / "results" / "a549" / "q2" / "q2_report.json").read_text())["results"]
    models = {}
    for label in LABELS:
        path = paths[f"virtual|{label}"]["path"]
        d = S.dataset("virtual", label, str(tmp / "cal.csv"))
        fitted = M.fit_fold(path, d["task"], d["X"], d["y"], d["plate"], d["is_dmso"], d["groups"], d["names"], d["ref"], seed=0)
        models[label] = {"fitted": fitted, "path": path, "names": d["names"], "task": d["task"], "n": len(d["y"]),
                         "threshold": fitted.threshold, "has_proba": fitted.has_proba()}
    return models, sites


def q2(report):
    import sklearn
    from sklearn.metrics import r2_score, roc_auc_score
    from sweep.q2_toxicity import load_wells
    models, sites = fit_q2()
    tmp = Path(tempfile.mkdtemp())
    held = load_wells(str(_held_csv(sites, tmp)), 0.5)
    held_dmso = (held.pert_iname == "DMSO").to_numpy()
    for label in LABELS:
        m = models[label]
        path, fitted, d = m["path"], m["fitted"], {"names": m["names"], "task": m["task"], "y": range(m["n"])}
        # held-out plate: same labels as in the search (from the real stains), never used for fitting
        X = held[d["names"]].to_numpy(float)
        if label == "treated_vs_dmso":
            rows = (~held_dmso & (held.viability >= 0.8).to_numpy()) | held_dmso
            y = (~held_dmso[rows]).astype(int)
        else:
            rows = ~held_dmso
            y = held.viability.to_numpy()[rows] if d["task"] == "reg" else (held.viability.to_numpy()[rows] < float(label.split("_")[1])).astype(int)
        plate = np.ones(len(held), int)  # plate 1 = the new plate: robust z uses ITS DMSO wells (label-free)
        if path["cleaning"] == "plate_robust_z_vs_dmso":
            Dm = X[held_dmso]
            med = np.median(Dm, 0)
            mad = 1.4826 * np.median(np.abs(Dm - med), 0)
            fitted.pre.clean_.stats[1] = (med, np.where(mad < 1e-12, Dm.std(0) + 1e-12, mad))
        s = fitted.score(X[rows], plate[rows])
        held_score = float(r2_score(y, s)) if d["task"] == "reg" else float(roc_auc_score(y, s))
        if path["cleaning"] == "plate_robust_z_vs_dmso":
            fitted.pre.clean_.stats.pop(1)
        report["q2"][label] = {"path": path, "fitted_on": f"{CAL}: {len(d['y'])} wells",
                               "held_out": f"{HELD}: {int(rows.sum())} wells",
                               ("held_out_R2" if d["task"] == "reg" else "held_out_AUC"): round(held_score, 3)}
        print(f"Q2 {label:16} fitted on {len(d['y'])} wells of {CAL}; held-out {HELD} "
              f"{'R2' if d['task'] == 'reg' else 'AUC'} {held_score:.3f}", flush=True)
    blob = cloudpickle.dumps({"sklearn": sklearn.__version__, "models": models})
    (OUT / "q2_final.pkl").write_bytes(blob)
    print("models/q2_final.pkl SHA-256 (set Q2_SHA256 in vcp_system.py to this):", hashlib.sha256(blob).hexdigest())


def _held_csv(sites, tmp):
    sites[sites.plate == HELD].to_csv(tmp / "held.csv", index=False)
    return tmp / "held.csv"


def q3(report, raw):
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score
    from sweep.q3_search import SHAPE, assign, fit_states, paired_idx
    cells = pd.read_parquet(Path(raw) / "q3_v2" / "cells.parquet")
    path = json.loads((ROOT / "results" / "a549" / "q3" / "q3_report.json").read_text())["states"]["path"]
    assert path["features"] == "morphology" and path["processing"] == "robust_z_vs_dmso" and path["embedding"] == "pca"
    cal = cells[cells.plate == CAL].reset_index(drop=True)
    real = cal[cal.source == "real"]
    X = np.nan_to_num(real[SHAPE].to_numpy(float))
    ref = X[(real.pert_iname == "DMSO").to_numpy()]
    med = np.median(ref, 0)
    mad = 1.4826 * np.median(np.abs(ref - med), 0) + 1e-9
    Z = np.clip((X - med) / mad, -20, 20)
    pca = PCA(min(10, Z.shape[1]), random_state=0).fit(Z)
    cent = fit_states(pca.transform(Z).astype(np.float32), path, np.random.default_rng(0))
    st = assign(pca.transform(Z).astype(np.float32), cent)
    desc = real.assign(state=st).groupby("state")[["area", "eccentricity", "solidity"]].mean().round(2)
    desc["share_of_cells"] = (pd.Series(st).value_counts(normalize=True).sort_index()).round(3).to_numpy()

    def states_of(df):
        A = np.clip((np.nan_to_num(df[SHAPE].to_numpy(float)) - med) / mad, -20, 20)
        return assign(pca.transform(A).astype(np.float32), cent)
    held = cells[cells.plate == HELD].reset_index(drop=True)
    lab = states_of(held)
    ri, vi = paired_idx(held)
    ari = float(adjusted_rand_score(lab[ri], lab[vi]))
    model = {"path": path, "cols": SHAPE, "med": med, "mad": mad, "pca_mean": pca.mean_, "pca_components": pca.components_,
             "centroids": cent,  # numbers only: loads with any library version
             "states": desc.reset_index().to_dict("records"),
             "seg_path": json.loads((ROOT / "results" / "a549" / "q3" / "seg_fields.json").read_text())["seg_path"]}
    (OUT / "q3_states.json").write_text(json.dumps({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in model.items()}))
    report["q3"] = {"path": path, "n_states": int(len(cent)), "fitted_on": f"{CAL}: {len(real)} real cells",
                    "held_out": f"{HELD}: {len(ri)} matched real/virtual cell pairs",
                    "held_out_ARI_real_vs_virtual_states": round(ari, 3), "states": model["states"]}
    print(f"Q3 states: {len(cent)} states fitted on {len(real)} real cells of {CAL}; held-out {HELD} ARI {ari:.3f}")
    print(desc.to_string())


def q4(report, raw, chips):
    from sklearn.metrics import roc_auc_score
    from sweep import q4_chip_qc as Q
    emb = Path(raw) / "kaggle" / "vcp-q4-embeddings-and-finetune" / "q4" / "emb"
    data = Q.Data(emb)
    a = json.loads((ROOT / "results" / "chips" / "q4" / "q4_assess.json").read_text())
    prep, rep, fus, red, model = a["best"].split("/")
    path = {"preprocessing": prep, "representation": rep, "fusion": fus, "reduction": red, "model": model}
    X, y, g, cell = data.features(path), data.y, data.g, data.cell
    rng = np.random.default_rng(0)
    held_exp = []
    for c in sorted(set(cell)):  # one distinct experiment per cell type, with both good and bad chips if possible
        exps = [e for e in sorted(set(g[cell == c])) if e not in held_exp]
        both = [e for e in exps if len(set(y[g == e])) == 2]
        held_exp.append(str(rng.choice(both or exps)))
    held = np.isin(g, held_exp)
    pipe = Q.build(path, int((~held).sum()), X.shape[1], proba=True).fit(X[~held], y[~held])
    p = pipe.predict_proba(X[held])[:, 1]
    auc = float(roc_auc_score(y[held], p))
    idx = data.idx[held].assign(p_good=p)
    # the example images (held-out experiments only) for the app and the video
    from sweep.q4_parts import build_index
    files = build_index(Path(chips).parent / "OOC_datasheet.xlsx", img_dir=chips)[["imageID", "src"]]
    idx = idx.merge(files, on="imageID", how="left")
    imp, sc, cf, lr = pipe.steps[0][1], pipe.steps[1][1], pipe.steps[2][1], pipe.steps[3][1]
    model_ = {"path": path, "held_out_experiments": held_exp,  # numbers only: loads with any library version
              "impute": imp.statistics_.tolist(), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
              "keep": cf.keep_.tolist(), "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}
    z = ((np.where(np.isnan(X[held]), np.array(model_["impute"]), X[held]) - sc.mean_) / sc.scale_)[:, cf.keep_]
    p_np = 1 / (1 + np.exp(-(z @ lr.coef_[0] + lr.intercept_[0])))
    assert np.allclose(p_np, p, atol=1e-6), "numpy re-implementation differs from the pipeline"
    (OUT / "q4_chip_qc.json").write_text(json.dumps(model_))
    # examples: 3 random good + 3 random bad held-out images (seed 0), drawn without looking at the predictions
    ex = pd.concat([idx[idx.good == 1].sample(3, random_state=0), idx[idx.good == 0].sample(3, random_state=0)])
    report["q4"] = {"path": path, "examples_chosen": "3 random good + 3 random bad held-out images (seed 0), drawn without looking at the predictions", "fitted_on": f"{int((~held).sum())} images, {len(set(g[~held]))} experiments",
                    "held_out": f"{int(held.sum())} images, experiments {held_exp} (one per cell type)",
                    "held_out_AUC": round(auc, 3),
                    "examples": [{"imageID": r.imageID, "cell": r.cell, "good": int(r.good), "p_good": round(float(r.p_good), 3),
                                  "file": str(Path(r.src).relative_to(chips)) if isinstance(r.src, str) else None}
                                 for r in ex.itertuples()]}
    idx[["imageID", "experiment", "cell", "good", "p_good"]].to_csv(OUT / "q4_held_out_predictions.csv", index=False)
    print(f"Q4 fitted on {int((~held).sum())} images; held-out {held_exp} ({int(held.sum())} images) AUC {auc:.3f}")
    return emb, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT.parents[1] / "raw_runs_archive" / "runs"),
                    help="raw run outputs (q3_v2/cells.parquet, Q4 embeddings)")
    ap.add_argument("--chips", default=str(ROOT.parents[2] / "data" / "external" / "ooc" / "OOC_image_dataset"))
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    report = {"q2": {}, "note": "Final models of the winning paths (nothing re-chosen); held-out = data never used to fit them."}
    q2(report)
    q3(report, a.raw)
    q4(report, a.raw, a.chips)
    (OUT / "final_models.json").write_text(json.dumps(report, indent=1, default=str))
    print("saved", OUT)


if __name__ == "__main__":
    main()
