"""Collect every number used in the report / Writeup from results/ into results/results.json.

Missing results (a job still running) are simply absent; build_report.py shows them as "pending".
    python tools/collect_results.py
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
R = {}


def f3(x):
    return None if x is None else f"{x:.3f}"


def ci(d):
    return f"{d['estimate']:.3f} [{d['ci_low']:.3f}, {d['ci_high']:.3f}]"


def pval(p):
    return f"{p:.2g}" if p >= 1e-4 else f"{p:.1e}"


# ------------------------------------------------------------------------------------------ Q1 search
def q1():
    db = RES / "a549" / "q1_search" / "study.db"
    if db.exists():
        import optuna
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        st = optuna.load_study(study_name="q1", storage=f"sqlite:///{db}")
        allt = [t for t in st.trials if t.intermediate_values]
        tr = [t for t in allt if "u2os" not in t.params.get("data", "")]  # A549 study only (datasets are not path nodes)
        comp = [t for t in tr if t.state.name == "COMPLETE"]
        best = max(comp, key=lambda t: t.value)
        base = [t for t in comp if t.params.get("width") == 32 and t.params.get("depth") == 4
                and t.params.get("upsampling") == "transpose" and t.params.get("data") == "brightfield_a549"
                and t.params.get("processing") == "plate_scaling" and t.params.get("model") == "unet"]
        distinct = {tuple(sorted(t.params.items())) for t in tr}
        R.update(q1_excluded_mixed=len(allt) - len(tr), q1_paths=len(tr), q1_completed=len(comp), q1_stopped=sum(t.state.name == "PRUNED" for t in tr),
                 q1_distinct=len(distinct), q1_duplicates=len(tr) - len(distinct),
                 q1_best_val=f"{best.value:.4f}", q1_best_params=best.params,
                 q1_baseline_val=f"{max(t.value for t in base):.4f}" if base else None,
                 q1_best_gap=f"{best.user_attrs.get('p3_gap', float('nan')):.4f}")
    fs = RES / "a549" / "q1_final" / "final_summary.csv"
    if fs.exists():
        d = pd.read_csv(fs)
        R["q1_final_table"] = d.to_dict("records")
        R["q1_single_seed"] = ", ".join(d[d["count"] < 2].config.str.replace("t038_brightfield_", "").tolist()) or "none"
    w = RES / "a549" / "q1_final" / "q1_winner.json"
    if w.exists():
        w = json.loads(w.read_text())
        R.update(q1_winner=w["config"], q1_inference=w["inference"],
                 q1_inference_val={k: f"{v:.4f}" for k, v in w["inference_val_r"].items()})
    t = RES / "a549" / "q1_test" / "report.json"
    if t.exists():
        q1_test_values(json.loads(t.read_text()), "q1_test")


def q1_test_values(t, pre):
    if True:  # A549 and U2OS test reports
        pix = pd.DataFrame(t["pixel"])
        held = pix[pix.group == "held-out cpds"].set_index("channel")
        R[f"{pre}_pixel_r"] = {c: f"{held.loc[c, 'pearson']:.3f}" for c in held.index}
        a = t["phenotypic_activity"]
        if isinstance(a, dict) and a:
            R[f"{pre}_map"] = {s: a[s]["all"]["mAP"] for s in a}
            R[f"{pre}_map_heldout"] = {s: a[s]["held-out cpds"]["mAP"] for s in a}
            R[f"{pre}_frac_active"] = {s: a[s]["all"]["frac_active"] for s in a}
        ag = t["nuclei_agreement"]["all"]
        R.update(**{f"{pre}_nuclei_r": ci(ag["pearson"]), f"{pre}_nuclei_ccc": ci(ag["lins_ccc"]), f"{pre}_nuclei_icc": f3(ag["icc_2_1"]),
                 f"{pre}_ba_bias": f"{ag['bland_altman']['bias']:.2f}",
                 f"{pre}_ba_loa": f"[{ag['bland_altman']['loa_low']:.1f}, {ag['bland_altman']['loa_high']:.1f}]",
                 f"{pre}_slope": f"{ag['regression']['slope']:.3f} [{ag['regression']['slope_ci'][0]:.3f}, {ag['regression']['slope_ci'][1]:.3f}]"})


# ------------------------------------------------------------------------------------------ Q2
def q2(d=None, pre="q2"):
    d = d or RES / "a549" / "q2"  # rerun on the Q1 winner's virtual stains (real / bright-field searches unchanged)
    f = d / "q2_report.json"
    if not f.exists():
        return
    rep = json.loads(f.read_text())
    for key, r in rep["results"].items():
        src, lab = key.split("|")
        k = f"{pre}_{lab}_{src}".replace("+", "_").replace(".", "")
        if lab == "viability":
            R[k] = f"{r['r2']:.3f}"
            R[k + "_ccc"] = ci(r["agreement"]["lins_ccc"])
            R[k + "_ba"] = f"{r['agreement']['bland_altman']['bias']:+.3f} [{r['agreement']['bland_altman']['loa_low']:.3f}, {r['agreement']['bland_altman']['loa_high']:.3f}]"
        else:
            R[k] = ci(r["classification"]["auc"])
            R[k + "_lopo"] = "/".join(f"{x:.3f}" for x in r["leave_one_plate_out_auc"]) or "n/a (one plate)"
            R[k + "_perm"] = pval(r["permutation_p"])
            b = r.get("bayes_laplace_auc")
            R[k + "_bayes"] = ci(b) if isinstance(b, dict) else None
        R[k + "_path"] = " → ".join(str(v) for v in r["path"].values())
    for c in rep["comparisons"]:
        k = f"{pre}_cmp_{c['label']}_{c['comparison']}".replace(" vs ", "_vs_").replace("+", "_").replace(".", "")
        R[k] = {"diff": f"{c['diff']:+.3f}", "p": pval(c["p"]), "p_holm": pval(c["p_holm"]),
                "p_bh": pval(c["p_bh"]), "bayes_p": f"{c['bayes']['p_gt_0']:.3f}",
                "cri": f"[{c['bayes']['cri95'][0]:+.3f}, {c['bayes']['cri95'][1]:+.3f}]"}
    if (d / "searches").exists():
        ev, ex = 0, []
        for s_ in sorted((d / "searches").glob("*_summary.json")):
            j = json.loads(s_.read_text())
            ev += j["evaluated"]
            ex += [(e["node"], e["option"]) for e in j["excluded_options"]]
        R[f"{pre}_paths_total"] = ev
        R[f"{pre}_excluded_top"] = pd.Series([f"{n}={o}" for n, o in ex]).value_counts().head(6).to_dict()


# ------------------------------------------------------------------------------------------ Q3
def q3(d=None, pre="q3"):
    d = d or RES / "a549" / "q3"  # rerun on the Q1 winner's virtual stains
    for name, key in (("seg", "seg"), ("down", "down"), ("sup_virtual", "sup_virtual")):
        f = d / name / f"q3_{name}_summary.json"
        if f.exists():
            s_ = json.loads(f.read_text())
            R.update(**{f"{pre}_{key}_paths": s_["evaluated"], f"{pre}_{key}_completed": s_["completed_100pct"],
                        f"{pre}_{key}_excluded": [f"{e['node']}={e['option']}" for e in s_["excluded_options"]]})
            if key == "seg":
                R[f"{pre}_seg_best"] = f"{s_['best'][0]['value']:.3f}"
    f = d / "q3_report.json"
    if not f.exists():
        return
    r = json.loads(f.read_text())
    R.update(**{f"{pre}_seg_path": r["seg_path"], f"{pre}_f1": ci(r["f1"]), f"{pre}_ap": f"{r['ap_mean_0.5_0.95']:.3f}",
                f"{pre}_dice": f3(r["dice_matched"]), f"{pre}_count_ccc": ci(r["count_agreement"]["lins_ccc"]),
                f"{pre}_count_bias": f"{r['count_agreement']['bland_altman']['bias']:.2f}",
                f"{pre}_count_loa": f"[{r['count_agreement']['bland_altman']['loa_low']:.1f}, {r['count_agreement']['bland_altman']['loa_high']:.1f}]",
                f"{pre}_feature_ccc": f"{r['feature_ccc_median']:.3f}"})
    if "states" in r:
        s_ = r["states"]
        R.update(**{f"{pre}_states_path": " → ".join(s_["path"].values()), f"{pre}_n_states": s_["n_states"],
                    f"{pre}_ari": f"{s_['ari']:.3f}", f"{pre}_nmi": f"{s_['nmi']:.3f}",
                    f"{pre}_stability": f"{min(s_['bootstrap_stability_ari']):.2f}–{max(s_['bootstrap_stability_ari']):.2f}"})
    if "supervised_auc" in r:
        R[f"{pre}_sup"] = {k: f"{v:.3f}" for k, v in r["supervised_auc"].items()}
    R[f"{pre}_dist"] = {k: f"{v['wasserstein_over_sd_median']:.2f}" for k, v in r.get("per_compound_distance", {}).items()}


# ------------------------------------------------------------------------------------------ U2OS replication
def u2os():
    """Second dataset, separate study: the A549 winning paths (nothing re-chosen) on U2OS BR00117011 (train) / BR00117012 (test)."""
    d = RES / "u2os"
    w = d / "q1" / "q1_replicate_winner.json"
    if w.exists():
        w = json.loads(w.read_text())
        R["u2_val_r"] = " / ".join(f"{v:.3f}" for v in w["val_r"])
        R["u2_seeds"] = len(w["val_r"])
    t = d / "q1" / "report.json"
    if t.exists():
        q1_test_values(json.loads(t.read_text()), "u2_test")
    q2(d / "q2", "u2_q2")
    q3(d / "q3", "u2_q3")


# ------------------------------------------------------------------------------------------ Q4
def q4():
    d = RES / "chips" / "q4"
    if not (d / "q4_assess.json").exists():
        return
    a = json.loads((d / "q4_assess.json").read_text())
    s = json.loads((d / "q4_search_summary.json").read_text())
    rp = a["reports"]
    R.update(q4_paths=s["evaluated"], q4_completed=s["completed_100pct"],
             q4_excluded=[f"{e['node']}={e['option']}" for e in s["excluded_options"]],
             q4_best=ci(rp[a["best"]]["auc"]), q4_baseline=ci(rp[a["baseline"]]["auc"]),
             q4_best_brier=f3(rp[a["best"]]["brier"]),
             q4_loco=f"{a['leave_one_cell_type_out'][a['best']]['pooled_auc']:.3f}",
             q4_loco_base=f"{a['leave_one_cell_type_out'][a['baseline']]['pooled_auc']:.3f}",
             q4_lodo=f"{a['leave_one_day_out'][a['best']]['pooled_auc']:.3f}",
             q4_lodo_base=f"{a['leave_one_day_out'][a['baseline']]['pooled_auc']:.3f}",
             q4_per_cell={k: f"{v['estimate']:.3f}" for k, v in a["per_cell_type_auc"]["best"].items()},
             q4_finetune=f"{a['finetune']['cv_auc']:.3f}", q4_celltype_only=f"{a['cell_type_only_reference_auc']:.3f}")


if __name__ == "__main__":
    for fn in (q1, q2, q3, q4, u2os):
        fn()
    (RES / "results.json").write_text(json.dumps(R, indent=1, default=str))
    print(len(R), "values ->", RES / "results.json")
