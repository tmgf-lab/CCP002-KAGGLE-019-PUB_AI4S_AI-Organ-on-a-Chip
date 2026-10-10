"""Q2 guided search (sweep/configs/q2_toxicity.yaml): one engine Search per (source, label).

Path = cleaning -> transform -> selection -> reduction -> model -> calibration -> imbalance -> score.
Sources: real / virtual / brightfield / brightfield+virtual image features per well.
Labels : toxic_0.5, toxic_0.8 (compound wells), viability (regression, R^2), treated_vs_dmso
         (non-toxic compound wells, viability >= 0.8, vs DMSO wells).
Scored by grouped CV (compounds; each DMSO well its own group) on nested 25/50/100% subsets of groups.

    python -m sweep.q2_search search --source virtual --label treated_vs_dmso [--budget 250 --workers 4 --hours 4]
    python -m sweep.q2_search assess
    python -m sweep.q2_search all --workers 4 --sources real virtual --labels toxic_0.5 viability
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sweep import engine, stats  # noqa: E402
from sweep import q2_models as M  # noqa: E402
from sweep.q2_toxicity import LEAK, load_wells  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {"real": ("real_",), "virtual": ("virt_",), "brightfield": ("bf_",), "brightfield+virtual": ("bf_", "virt_")}
LABELS = ["toxic_0.5", "toxic_0.8", "viability", "treated_vs_dmso"]
COLOURS = {"real": "#2a78d6", "virtual": "#eb6834", "brightfield": "#1baf7a", "brightfield+virtual": "#4a3aa7"}
_DATA = {}


# ------------------------------------------------------------------------------------------ data
def dataset(source, label, features):
    key = (source, label, features)
    if key in _DATA:
        return _DATA[key]
    w = load_wells(features, 0.5)
    cols = [c for c in w.columns if c.startswith(SOURCES[source]) and not any(k in c for k in LEAK)]
    assert cols and not any(k in c for c in cols for k in LEAK)
    plates = sorted(w.plate.unique())  # one-plate datasets (U2OS replication): every well on plate 0
    plate = (w.plate == plates[1]).astype(int).to_numpy() if len(plates) > 1 else np.zeros(len(w), int)
    dmso = (w.pert_iname == "DMSO").to_numpy()
    if label == "treated_vs_dmso":
        rows = ((~dmso) & (w.viability >= 0.8).to_numpy()) | dmso
        y = (~dmso[rows]).astype(int)
        task, ref = "clf", None
    else:
        rows = ~dmso
        task = "reg" if label == "viability" else "clf"
        y = w.viability.to_numpy()[rows] if task == "reg" else (w.viability.to_numpy()[rows] < float(label.split("_")[1])).astype(int)
        ref = (w.loc[dmso, cols].to_numpy(float), plate[dmso])  # label-free DMSO reference for plate robust-z
    sub = w[rows]
    groups = np.where(sub.pert_iname == "DMSO", "DMSO_" + sub.plate + "_" + sub.well, sub.pert_iname).astype(str)
    d = {"X": sub[cols].to_numpy(float), "y": y, "plate": plate[rows], "is_dmso": dmso[rows], "groups": groups,
         "names": cols, "task": task, "ref": ref, "meta": sub[["plate", "well", "pert_iname", "split", "viability"]].reset_index(drop=True)}
    _DATA[key] = d
    return d


def cv_predict(path, d, idx, repeats, smoke=False, seed=0):
    """Out-of-fold scores on rows idx (grouped k-fold, repeated). Returns (mean score per fold-repeat, oof matrix)."""
    X, y, g = d["X"][idx], d["y"][idx], d["groups"][idx]
    pl, dm = d["plate"][idx], d["is_dmso"][idx]
    oofs, vals = [], []
    for r in range(repeats):
        splits = M.grouped_splits(y, g, 5, d["task"], seed + r)
        oof = np.full(len(y), np.nan)
        for tr, te in splits:
            fp = M.fit_fold(path, d["task"], X[tr], y[tr], pl[tr], dm[tr], g[tr], d["names"], d["ref"], smoke, seed + r)
            oof[te] = fp.score(X[te], pl[te])
        oofs.append(oof)
        vals.append(M.metric(y, oof, "reg") if d["task"] == "reg" else M._pooled_auc(y, oof, splits))
    return float(np.nanmean(vals)), np.vstack(oofs)


def score_fn(path, frac, trial=None):
    """Engine objective (module-level for spawn). Context comes from environment variables."""
    d = dataset(os.environ["Q2_SOURCE"], os.environ["Q2_LABEL"], os.environ["Q2_FEATURES"])
    smoke = os.environ.get("Q2_SMOKE") == "1"
    idx = engine.nested_subset(d["groups"], frac)
    s, _ = cv_predict(path, d, idx, 3 if (frac >= 1.0 and not smoke) else 1, smoke)
    return s if np.isfinite(s) else -1.0


def search_name(source, label):
    return f"q2_{source.replace('+', '_')}_{label}"


def run_search(source, label, a):
    os.environ.update(Q2_SOURCE=source, Q2_LABEL=label, Q2_FEATURES=str(a.features), Q2_SMOKE="1" if a.smoke else "0")
    task = "reg" if label == "viability" else "clf"
    tree = M.tree_for(source, task)
    s = engine.Search(search_name(source, label), tree, Path(a.out) / "searches", budget=a.budget,
                      baseline=M.baseline_for(task), warmup_per_option=1 if a.smoke else 3)
    t0 = time.time()
    summ = s.run(score_fn, workers=a.workers, hours=a.hours)
    print(f"{s.name}: evaluated {summ['evaluated']}, completed {summ['completed_100pct']}, "
          f"excluded {[(e['node'], e['option']) for e in summ['excluded_options']]}, "
          f"best {summ['best'][0]['value'] if summ['best'] else None}, {time.time() - t0:.0f}s", flush=True)
    return summ


# ------------------------------------------------------------------------------------------ assessment
def best_path(out, source, label):
    f = Path(out) / "searches" / f"{search_name(source, label)}_paths.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    df = df[(df.state == "COMPLETE") & df.skipped.isna()].sort_values("value", ascending=False)
    if not len(df):
        return None
    task = "reg" if label == "viability" else "clf"
    keys = list(M.tree_for(source, task))
    return {k: df.iloc[0][k] for k in keys}, float(df.iloc[0].value)


def bayes_bootstrap_p(y, s1, s2, groups, n=2000, seed=0):
    """Bayesian bootstrap (Dirichlet weights over groups): posterior of AUC(s1) - AUC(s2) and P(diff > 0)."""
    rng = np.random.default_rng(seed)
    ug, gi = np.unique(groups, return_inverse=True)
    diffs = []
    for _ in range(n):
        w = rng.dirichlet(np.ones(len(ug)))[gi]
        diffs.append(roc_auc_score(y, s1, sample_weight=w) - roc_auc_score(y, s2, sample_weight=w))
    diffs = np.array(diffs)
    return {"mean": float(diffs.mean()), "cri95": np.percentile(diffs, [2.5, 97.5]).round(4).tolist(),
            "p_gt_0": float((diffs > 0).mean())}


def laplace_logistic(Xtr, ytr, Xte, prior_sd=1.0):
    """Bayesian logistic regression, N(0, prior_sd^2) prior; Laplace approximation at the MAP; predictive via the
    probit approximation of E[sigmoid(x'w)] (MacKay 1992). Returns (mean prob, predictive sd of the logit)."""
    from scipy.optimize import minimize
    A = np.hstack([np.ones((len(Xtr), 1)), Xtr])
    prec = np.r_[1e-2, np.full(Xtr.shape[1], 1 / prior_sd ** 2)]

    def nlp(w):
        z = A @ w
        return np.sum(np.logaddexp(0, z) - ytr * z) + 0.5 * np.sum(prec * w ** 2), A.T @ (1 / (1 + np.exp(-z)) - ytr) + prec * w
    w = minimize(nlp, np.zeros(A.shape[1]), jac=True, method="L-BFGS-B").x
    s = 1 / (1 + np.exp(-(A @ w)))
    S = np.linalg.inv((A * (s * (1 - s))[:, None]).T @ A + np.diag(prec))
    B = np.hstack([np.ones((len(Xte), 1)), Xte])
    mu, var = B @ w, np.einsum("ij,jk,ik->i", B, S, B)
    return 1 / (1 + np.exp(-mu / np.sqrt(1 + np.pi * var / 8))), np.sqrt(var)


def laplace_auc(d, path, idx_splits):
    """Bayesian logistic (Laplace approximation, N(0,1) prior) on the best pipeline's preprocessed features: OOF AUC."""
    y = d["y"]
    oof = np.full(len(y), np.nan)
    for tr, te in idx_splits:
        pre = M.Preprocessor(path, "clf", d["names"], d["ref"]).fit(d["X"][tr], y[tr], d["plate"][tr], d["is_dmso"][tr])
        oof[te] = laplace_logistic(pre.transform(d["X"][tr], d["plate"][tr]), y[tr], pre.transform(d["X"][te], d["plate"][te]))[0]
    return oof


def assess(a):
    out = Path(a.out)
    res, oof_store, rows = {}, {}, []
    for label in LABELS:
        for source in SOURCES:
            bp = best_path(a.paths_from or out, source, label)  # replication: winning paths of another study
            if bp is None:
                continue
            path, val = bp
            d = dataset(source, label, a.features)
            allidx = np.arange(len(d["y"]))
            _, oofm = cv_predict(path, d, allidx, 1 if a.smoke else 3, a.smoke)
            oof = np.nanmean(oofm, 0)
            oof_store[(source, label)] = (d, oof)
            r = {"path": path, "search_value": val}
            if d["task"] == "clf":
                rep = stats.classification_report(d["y"], oof, groups=d["groups"])
                r.update(classification=rep, pr_auc=float(average_precision_score(d["y"], oof)))
                # label-permutation test of the OOF AUC (no refit): null distribution of AUC under shuffled labels
                rng = np.random.default_rng(0)
                null = np.array([roc_auc_score(rng.permutation(d["y"]), oof) for _ in range(200 if not a.smoke else 20)])
                r["permutation_p"] = float((np.sum(null >= rep["auc"]["estimate"]) + 1) / (len(null) + 1))
                # leave-one-plate-out
                lopo = []
                for test_pl in ((0, 1) if len(np.unique(d["plate"])) > 1 else ()):  # one plate: not possible
                    tr, te = np.where(d["plate"] != test_pl)[0], np.where(d["plate"] == test_pl)[0]
                    fp = M.fit_fold(path, "clf", d["X"][tr], d["y"][tr], d["plate"][tr], d["is_dmso"][tr], d["groups"][tr],
                                    d["names"], d["ref"], a.smoke)
                    s = fp.score(d["X"][te], d["plate"][te])
                    lopo.append(float(roc_auc_score(d["y"][te], s)))
                    if test_pl == 1:  # permutation importance on the held-out plate (top 15 raw features)
                        base = roc_auc_score(d["y"][te], s)
                        imp = {}
                        Xte = d["X"][te].copy()
                        for j in np.argsort(-np.abs(np.corrcoef(d["X"][tr].T, d["y"][tr])[-1, :-1]))[:15]:
                            Xp = Xte.copy()
                            Xp[:, j] = rng.permutation(Xp[:, j])
                            imp[d["names"][j]] = float(base - roc_auc_score(d["y"][te], fp.score(Xp, d["plate"][te])))
                        r["permutation_importance"] = dict(sorted(imp.items(), key=lambda kv: -kv[1]))
                r["leave_one_plate_out_auc"] = lopo
                # selected-feature stability across the 5 folds
                sel = {}
                for tr, te in M.grouped_splits(d["y"], d["groups"], 5, "clf", 0):
                    for f in M.Preprocessor(path, "clf", d["names"], d["ref"]).fit(
                            d["X"][tr], d["y"][tr], d["plate"][tr], d["is_dmso"][tr]).selected_features():
                        sel[f] = sel.get(f, 0) + 0.2
                r["selection_stability_top"] = dict(sorted(sel.items(), key=lambda kv: -kv[1])[:20])
                # Bayesian logistic (Laplace) on this pipeline's preprocessing
                try:
                    lb = laplace_auc(d, path, M.grouped_splits(d["y"], d["groups"], 5, "clf", 0))
                    r["bayes_laplace_auc"] = stats.bootstrap_ci(roc_auc_score, d["y"], lb, groups=d["groups"])
                except Exception as e:  # noqa: BLE001
                    r["bayes_laplace_auc"] = f"failed: {e}"
                auc = rep["auc"]
            else:
                r["agreement"] = stats.agreement_report(d["y"], oof, groups=d["groups"])
                r["r2"] = float(M.metric(d["y"], oof, "reg"))
                auc = {"estimate": r["r2"], "ci_low": np.nan, "ci_high": np.nan}
            res[f"{source}|{label}"] = r
            rows.append({"label": label, "source": source, "metric": "AUC" if d["task"] == "clf" else "R2",
                         "value": auc["estimate"], "ci_low": auc["ci_low"], "ci_high": auc["ci_high"],
                         "path": " -> ".join(str(v) for v in path.values())})
            print(f"  {label:<16} {source:<20} {rows[-1]['metric']} {auc['estimate']:.3f}", flush=True)

    # comparisons per classification label: DeLong (frequentist) + Bayesian bootstrap, Holm/BH over the family
    comps = []
    for label in [l for l in LABELS if l != "viability"]:
        for s1, s2 in [("virtual", "brightfield"), ("virtual", "real"), ("brightfield+virtual", "brightfield")]:
            if (s1, label) in oof_store and (s2, label) in oof_store:
                d1, o1 = oof_store[(s1, label)]
                d2, o2 = oof_store[(s2, label)]
                assert np.array_equal(d1["y"], d2["y"])
                dl = stats.delong(d1["y"], o1, o2)
                comps.append({"label": label, "comparison": f"{s1} vs {s2}", **dl,
                              "bayes": bayes_bootstrap_p(d1["y"], o1, o2, d1["groups"], n=300 if a.smoke else 2000)})
    if comps:
        from statsmodels.stats.multitest import multipletests
        ps = [c["p"] for c in comps]
        for c, h, b in zip(comps, multipletests(ps, method="holm")[1], multipletests(ps, method="fdr_bh")[1]):
            c["p_holm"], c["p_bh"] = float(h), float(b)
    table = pd.DataFrame(rows)
    table.to_csv(out / "q2_best_paths.csv", index=False)
    json.dump({"results": res, "comparisons": comps}, open(out / "q2_report.json", "w"), indent=1, default=str)
    figures(out, oof_store)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    for c in comps:
        print(f"  {c['label']:<16} {c['comparison']:<32} diff {c['diff']:+.3f} p {c['p']:.3g} holm {c['p_holm']:.3g} "
              f"P(>0|data) {c['bayes']['p_gt_0']:.3f}")


def figures(out, oof_store):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    labels = [l for l in LABELS if l != "viability" and any((s, l) in oof_store for s in SOURCES)]
    if labels:
        fig, axes = plt.subplots(1, len(labels), figsize=(4.6 * len(labels), 4.4), facecolor="#fcfcfb", squeeze=False)
        for ax, label in zip(axes[0], labels):
            ax.set_facecolor("#fcfcfb")
            ax.plot([0, 1], [0, 1], color="#b8b8b4", lw=1, ls="--")
            for s in SOURCES:
                if (s, label) in oof_store:
                    d, o = oof_store[(s, label)]
                    fpr, tpr, _ = roc_curve(d["y"], o)
                    ax.plot(fpr, tpr, color=COLOURS[s], lw=2, label=f"{s}: {roc_auc_score(d['y'], o):.3f}")
            ax.set(title=label, xlabel="False positive rate", ylabel="True positive rate")
            ax.legend(frameon=False, fontsize=8, loc="lower right")
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.savefig(out / "fig_q2_roc.png", dpi=170, facecolor=fig.get_facecolor())
        plt.close(fig)
    # convergence: best-so-far vs evaluated paths, per search
    files = sorted((out / "searches").glob("*_paths.csv"))
    if files:
        fig, axes = plt.subplots(1, len(LABELS), figsize=(4.4 * len(LABELS), 3.6), facecolor="#fcfcfb", squeeze=False)
        for ax, label in zip(axes[0], LABELS):
            for s in SOURCES:
                f = out / "searches" / f"{search_name(s, label)}_paths.csv"
                if not f.exists():
                    continue
                df = pd.read_csv(f)
                df = df[df.skipped.isna() & df.state.isin(["COMPLETE", "PRUNED"])].sort_values("trial")
                sc = df["score_100"].where(df.state == "COMPLETE")
                ax.plot(np.arange(1, len(df) + 1), sc.cummax().to_numpy(), color=COLOURS[s], lw=2, label=s)
            ax.set(title=label, xlabel="paths evaluated", ylabel="best score (100% data)")
            ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "fig_q2_convergence.png", dpi=150, facecolor=fig.get_facecolor())
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["search", "assess", "all"])
    ap.add_argument("--source", choices=list(SOURCES))
    ap.add_argument("--label", choices=LABELS)
    ap.add_argument("--sources", nargs="+", default=list(SOURCES))
    ap.add_argument("--labels", nargs="+", default=LABELS)
    ap.add_argument("--budget", type=int, default=250)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--hours", type=float, default=4.0)
    ap.add_argument("--features", default=os.environ.get("Q2_FEATURES_PATH", str(ROOT / "results/a549/q1_test/site_features.csv")))
    ap.add_argument("--out", default=os.environ.get("Q2_OUT", str(ROOT / "runs/q2")))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--paths-from", help="assess: use the best paths of the searches in this folder (replication study)")
    a = ap.parse_args()
    if a.smoke:
        a.budget = min(a.budget, 10)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    if a.command == "search":
        run_search(a.source, a.label, a)
    elif a.command == "assess":
        assess(a)
    else:
        for label in a.labels:
            for source in a.sources:
                run_search(source, label, a)
        assess(a)


if __name__ == "__main__":
    main()
