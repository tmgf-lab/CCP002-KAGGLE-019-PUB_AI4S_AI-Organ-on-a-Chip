"""Q2 toxicity: can virtual-stain features flag cytotoxic compounds as well as real-stain features?

Label (from the REAL stain, per well): viability = real_n_nuclei / median(real_n_nuclei of DMSO wells, same plate);
toxic = viability < 0.5. Features containing n_nuclei / nuc_area are dropped from every source (leakage guard).
Paths follow sweep/configs/q2_toxicity.yaml. CV is grouped by compound, 5-fold x repeats, inner 3-fold grouped CV.

    python -m sweep.q2_toxicity [--features results/a549/q1_test/site_features.csv] [--out runs/q2_toxicity] [--smoke]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sweep import stats

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "sweep" / "configs" / "q2_toxicity.yaml"
META = ["plate", "well", "site", "pert_iname", "split"]
SOURCES = {"real_features.csv": ("real", "real_"), "virtual_features.csv": ("virtual", "virt_"),
           "brightfield_features.csv": ("brightfield", "bf_")}
LEAK = ("n_nuclei", "nuc_area")
COLOURS = {"real": "#2a78d6", "virtual": "#eb6834", "brightfield": "#1baf7a"}
SEED = 0
N_OUTER, N_INNER, N_REPEATS = 5, 3, 3
GRIDS = {"logistic_regression": {"logisticregression__C": [0.1, 1, 10]},
         "gradient_boosting": {"histgradientboostingclassifier__learning_rate": [0.03, 0.1]}}
RIDGE_ALPHAS = [0.1, 1, 10, 100, 1000]


# ------------------------------------------------------------------------------------------ data
def load_wells(path, threshold=0.5):
    sites = pd.read_csv(path)
    feats = [c for c in sites.columns if c not in META]
    wells = sites.groupby(["plate", "well"], as_index=False).agg(
        {"pert_iname": "first", "split": "first", **{c: "mean" for c in feats}})
    dmso_med = wells[wells.pert_iname == "DMSO"].groupby("plate")["real_n_nuclei"].median()
    wells["viability"] = wells["real_n_nuclei"] / wells["plate"].map(dmso_med)
    wells["toxic"] = (wells["viability"] < threshold).astype(int)
    return wells


def feature_columns(wells, prefix):
    cols = [c for c in wells.columns if c.startswith(prefix) and not any(k in c for k in LEAK)]
    assert cols and not any(k in c for c in cols for k in LEAK)
    return cols


def parse_paths(cfg):
    out = []
    for p in cfg["paths"]:
        data, proc, model, ev = [s.strip() for s in p.split("->")]
        out.append({"path": f"{data} -> {proc} -> {model} -> {ev}", "data": data, "processing": proc,
                    "model": model, "source": SOURCES[data][0], "prefix": SOURCES[data][1]})
    return out


# ------------------------------------------------------------------------------------------ models
def build_pipeline(processing, model):
    steps = [StandardScaler()]
    if processing == "pca":
        steps.append(PCA(n_components=0.95, svd_solver="full", random_state=SEED))
    if model == "logistic_regression":
        steps.append(LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000))
    else:
        steps.append(HistGradientBoostingClassifier(learning_rate=0.1, random_state=SEED))
    return make_pipeline(*steps)


def permuted_groups(groups, seed):
    """Shuffle fold assignment by relabelling compounds with a random permutation of integer ids."""
    uniq = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    return pd.Series(groups).map(dict(zip(uniq, perm))).to_numpy()


def inner_select(pipe, grid, X, y, g, inner, seed):
    """Pick the grid point with the best mean inner grouped-CV AUC."""
    key, values = next(iter(grid.items()))
    if not inner:  # smoke: defaults (C=1, learning_rate=0.1)
        return clone(pipe)
    cv = StratifiedGroupKFold(n_splits=N_INNER, shuffle=False)
    gp = permuted_groups(g, seed)
    scores = []
    for v in values:
        s = []
        for tr, te in cv.split(X, y, gp):
            m = clone(pipe).set_params(**{key: v}).fit(X[tr], y[tr])
            s.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        scores.append(np.mean(s))
    return clone(pipe).set_params(**{key: values[int(np.argmax(scores))]})


def oof_classify(X, y, g, processing, model, repeats, inner):
    pipe, grid = build_pipeline(processing, model), GRIDS[model]
    preds = np.zeros((repeats, len(y)))
    for r in range(repeats):
        gp = permuted_groups(g, SEED + r)
        for k, (tr, te) in enumerate(StratifiedGroupKFold(n_splits=N_OUTER, shuffle=False).split(X, y, gp)):
            assert not set(g[tr]) & set(g[te])
            m = inner_select(pipe, grid, X[tr], y[tr], g[tr], inner, SEED + 100 * r + k)
            preds[r, te] = m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return preds.mean(0)


def oof_ridge(X, v, g, repeats, inner):
    preds = np.zeros((repeats, len(v)))
    for r in range(repeats):
        gp = permuted_groups(g, SEED + r)
        for k, (tr, te) in enumerate(GroupKFold(n_splits=N_OUTER).split(X, v, gp)):
            alpha = 10.0
            if inner:
                gi = permuted_groups(g[tr], SEED + 100 * r + k)
                errs = []
                for a in RIDGE_ALPHAS:
                    e = []
                    for itr, ite in GroupKFold(n_splits=N_INNER).split(X[tr], v[tr], gi):
                        m = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X[tr][itr], v[tr][itr])
                        e.append(np.mean((m.predict(X[tr][ite]) - v[tr][ite]) ** 2))
                    errs.append(np.mean(e))
                alpha = RIDGE_ALPHAS[int(np.argmin(errs))]
            m = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(X[tr], v[tr])
            preds[r, te] = m.predict(X[te])
    return preds.mean(0)


# ------------------------------------------------------------------------------------------ figure
def plot_roc(y, best, reports, out, title="Toxicity"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 5), facecolor="#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.plot([0, 1], [0, 1], color="#b8b8b4", lw=1, ls="--", zorder=1)
    for src in ["real", "virtual", "brightfield"]:
        path, p = best[src]
        fpr, tpr, _ = roc_curve(y, p)
        a = reports[path]["auc"]
        ax.plot(fpr, tpr, color=COLOURS[src], lw=2, zorder=2,
                label=f"{src}: AUC {a['estimate']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate", ylabel="True positive rate",
           title=f"{title}, best path per source\nn={len(y)} wells, {int(y.sum())} positive")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)


# ------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "results/a549/q1_test/site_features.csv"))
    ap.add_argument("--out", default=str(ROOT / "runs/q2_toxicity"))
    ap.add_argument("--smoke", action="store_true", help="1 repeat, no inner CV")
    ap.add_argument("--task", choices=["toxic", "treated_vs_dmso"], default="toxic",
                    help="toxic: viability < threshold among compound wells; treated_vs_dmso: NON-toxic compound "
                         "wells (viability >= threshold) vs DMSO wells - does the source see a drug effect beyond cell death?")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()
    t0 = time.time()
    repeats, inner = (1, False) if args.smoke else (N_REPEATS, True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wells = load_wells(args.features, args.threshold)
    if args.task == "toxic":
        cw = wells[wells.pert_iname != "DMSO"].reset_index(drop=True)
        cw["label"] = cw["toxic"]
        groups = cw["pert_iname"].to_numpy()
        label_text = f"toxic = real_n_nuclei / plate DMSO median < {args.threshold}"
        title = f"Toxicity (viability < {args.threshold})"
    else:
        treated = wells[(wells.pert_iname != "DMSO") & (wells.viability >= args.threshold)]
        dmso = wells[wells.pert_iname == "DMSO"]
        cw = pd.concat([treated, dmso]).reset_index(drop=True)
        cw["label"] = (cw.pert_iname != "DMSO").astype(int)
        # each DMSO well is its own group; a compound's wells (both plates) always share a fold
        groups = np.where(cw.pert_iname == "DMSO", "DMSO_" + cw.plate + "_" + cw.well, cw.pert_iname)
        label_text = f"treated (non-toxic: viability >= {args.threshold}) = 1, DMSO = 0"
        title = f"Non-toxic treated vs DMSO (viability >= {args.threshold})"
    y, g, v = cw["label"].to_numpy(), groups, cw["viability"].to_numpy()
    print(f"task {args.task}: wells {len(cw)}, groups {len(np.unique(g))}, positive {y.sum()} ({y.mean():.1%})")

    paths = parse_paths(yaml.safe_load(open(CONFIG)))
    oof = cw[["plate", "well", "pert_iname", "split", "viability", "label"]].copy()
    reports, rows = {}, []
    for P in paths:
        cols = feature_columns(cw, P["prefix"])
        p = oof_classify(cw[cols].to_numpy(float), y, g, P["processing"], P["model"], repeats, inner)
        oof[P["path"]] = p
        rep = stats.classification_report(y, p, groups=g)
        rep["n_features"] = len(cols)
        reports[P["path"]] = rep
        rows.append({"path": P["path"], "source": P["source"], "auc": rep["auc"]["estimate"],
                     "auc_ci_low": rep["auc"]["ci_low"], "auc_ci_high": rep["auc"]["ci_high"],
                     "brier": rep["brier"], "n": rep["n"], "n_pos": int(y.sum()), "n_features": len(cols)})
        print(f"  {P['path']:<70} AUC {rep['auc']['estimate']:.3f}  ({time.time() - t0:.0f}s)")
    table = pd.DataFrame(rows)
    table.to_csv(out / "paths.csv", index=False)

    best = {}
    for src in ["real", "virtual", "brightfield"]:
        r = table[table.source == src].sort_values("auc", ascending=False).iloc[0]
        best[src] = (r["path"], oof[r["path"]].to_numpy())
    dl = {"virtual_vs_brightfield": {"path_1": best["virtual"][0], "path_2": best["brightfield"][0],
                                     **stats.delong(y, best["virtual"][1], best["brightfield"][1])},
          "virtual_vs_real": {"path_1": best["virtual"][0], "path_2": best["real"][0],
                              **stats.delong(y, best["virtual"][1], best["real"][1])}}

    viab = {}
    for data, (src, prefix) in (SOURCES.items() if args.task == "toxic" else []):
        pv = oof_ridge(cw[feature_columns(cw, prefix)].to_numpy(float), v, g, repeats, inner)
        oof[f"viability_pred_{src}"] = pv
        viab[src] = stats.agreement_report(v, pv, groups=g)
    oof.to_csv(out / "oof_predictions.csv", index=False)

    plot_roc(y, best, reports, out / "fig_q2_roc.png", title)
    # sanity: how hard is the task? best single (sign-free) feature AUC per source, and AUC on held-out compounds
    single = {}
    for data, (src, prefix) in SOURCES.items():
        a = {c: roc_auc_score(y, cw[c]) for c in feature_columns(cw, prefix)}
        k = max(a, key=lambda c: abs(a[c] - 0.5))
        single[src] = {"feature": k, "auc": float(max(a[k], 1 - a[k]))}
    held = ((cw["split"] == "test") | (cw.pert_iname == "DMSO")).to_numpy() if args.task != "toxic" \
        else (cw["split"] == "test").to_numpy()
    heldout = {s: {"path": best[s][0], "n": int(held.sum()), "n_pos": int(y[held].sum()),
                   **stats.bootstrap_ci(roc_auc_score, y[held], best[s][1][held], groups=g[held])} for s in best}
    summary = {
        "task": args.task, "threshold": args.threshold,
        "n_wells": int(len(y)), "n_groups": int(len(np.unique(g))), "n_positive": int(y.sum()),
        "prevalence": float(y.mean()), "smoke": args.smoke, "repeats": repeats, "inner_cv": inner,
        "best_auc": {s: {"path": best[s][0], **reports[best[s][0]]["auc"]} for s in best},
        "delong_p": {k: d["p"] for k, d in dl.items()},
        "viability_pearson": {s: viab[s]["pearson"]["estimate"] for s in viab},  # toxic task only
        "viability_ccc": {s: viab[s]["lins_ccc"]["estimate"] for s in viab},
        "best_single_feature_auc": single, "heldout_compound_auc": heldout,
        "runtime_s": round(time.time() - t0, 1)}
    json.dump({"summary": summary, "classification": reports, "delong": dl, "viability_agreement": viab,
               "label": label_text,
               "leakage_guard": f"dropped columns containing {list(LEAK)}"},
              open(out / "report.json", "w"), indent=2)

    print("\n" + table[["path", "auc", "auc_ci_low", "auc_ci_high", "brier", "n"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for k, d in dl.items():
        print(f"DeLong {k}: {d['auc_1']:.3f} vs {d['auc_2']:.3f}, diff {d['diff']:+.3f}, p={d['p']:.3g}")
    for s, r in viab.items():
        print(f"viability {s:<11} pearson {r['pearson']['estimate']:.3f} "
              f"[{r['pearson']['ci_low']:.3f},{r['pearson']['ci_high']:.3f}]  CCC {r['lins_ccc']['estimate']:.3f}")
    for s_ in single:
        print(f"sanity {s_:<11} best single feature {single[s_]['feature']} AUC {single[s_]['auc']:.3f}; "
              f"held-out compounds AUC {heldout[s_]['estimate']:.3f} (n={heldout[s_]['n']}, pos={heldout[s_]['n_pos']})")
    print(f"runtime {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
