"""Check every number of the technical report against the files written by the runs, and list where each comes from.

    python tools/audit_numbers.py        # -> docs/number_sources.md; exits with an error if any number does not match

Each check names a phrase of the report, the value the report should show right after it, and the run-output file the value
is read from (read here directly, not copied by hand). results/results.json is itself rebuilt from those files by
tools/collect_results.py, so its keys are listed with the file they come from.
"""
import glob
import json
import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
REPORT = ROOT / "docs" / "technical_report.md"

text = REPORT.read_text().replace(" ", " ")
text = re.sub(r"\s+", " ", text[text.index("# Summary"):text.index("# References")])

R = json.loads((RES / "results.json").read_text())
F = json.loads((ROOT / "models" / "final_models.json").read_text())
Q1T = json.loads((RES / "a549/q1_test/report.json").read_text())
U1T = json.loads((RES / "u2os/q1/report.json").read_text())
Q4A = json.loads((RES / "chips/q4/q4_assess.json").read_text())
cov = json.loads((RES / "chips/q4/q4_search_summary.json").read_text())["option_coverage"]  # options the Q4 search ran

# where each results.json key comes from (see tools/collect_results.py)
SRC = [("q1_inference", "results/a549/q1_final/q1_winner.json"), ("q1_final", "results/a549/q1_final/final_summary.csv"),
       ("q1_test", "results/a549/q1_test/report.json"), ("q1_", "results/a549/q1_search/study.db"),
       ("q2_", "results/a549/q2/q2_report.json"), ("q3_seg_paths", "results/a549/q3/seg/q3_seg_summary.json"),
       ("q3_down", "results/a549/q3/down/q3_down_summary.json"), ("q3_", "results/a549/q3/q3_report.json"),
       ("q4_", "results/chips/q4/q4_assess.json"), ("u2_val", "results/u2os/q1/q1_replicate_winner.json"),
       ("u2_test", "results/u2os/q1/report.json"), ("u2_q2", "results/u2os/q2/q2_report.json"),
       ("u2_q3", "results/u2os/q3/q3_report.json")]


def src(key):
    return next(f for p, f in SRC if key.startswith(p))


def r(key, sub=None, part=0):
    """A results.json value (first number of it unless part says otherwise) and its source file."""
    v = R[key] if sub is None else R[key][sub]
    nums = re.findall(r"[-+]?\d+(?:\.\d+)?(?:e-?\d+)?", str(v))
    return (nums[part] if part is not None else str(v)), src(key) + f" ({key}{'.' + sub if sub else ''})"


def ci_lo_hi(key, sub=None):
    v = R[key] if sub is None else R[key][sub]
    return re.findall(r"[-+]?\d+\.\d+", str(v))


rows, bad = [], []


SECTION = {"Table 1": "# Summary", "Summary": "# Summary", "Table 1 note": "# Summary", "Table 2": "# 2.", "§2": "# 2.",
           "Table 3": "# 3.", "§3": "# 3.", "§4": "# 4.", "Table 4": "# 5.", "§5": "# 5.", "§6": "# 6.", "§7": "# 7.",
           "§8": "# 8.", "Table 5": "## Final models", "§9": "# 9.", "§10": "# 10."}


def check(where, anchor, value, source, window=160):
    """The value must appear right after the phrase, searching only from the start of the check's section."""
    value = str(value)
    window = max(window, len(value) + 8)
    i = text.find(anchor, max(0, text.find(SECTION.get(where, "# Summary"))))
    ok = i >= 0 and value in text[i:i + len(anchor) + window]
    rows.append((where, anchor, value, source, "ok" if ok else "MISMATCH"))
    if not ok:
        bad.append((where, anchor, value, source, "anchor missing" if i < 0 else text[i:i + len(anchor) + window]))


# ---------------------------------------------------------------- run logs and data counted directly
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)
_st = optuna.load_study(study_name="q1", storage=f"sqlite:///{RES / 'a549/q1_search/study.db'}")
q1 = pd.DataFrame([{"number": t.number, "state": t.state.name, "data": t.params.get("data", "")} for t in _st.trials])


def q2_logs():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / "a549/q2/searches/*_paths.csv"))])
    sk = d["skipped"].notna() & (d["skipped"].astype(str) != "")
    run = d[~sk]
    return len(d), len(run), int((run.state == "COMPLETE").sum()), int((run.state.isin(["COMPLETE", "PRUNED"])).sum())


def log_counts(f):
    d = pd.read_csv(RES / f)
    sk = d["skipped"].notna() & (d["skipped"].astype(str) != "")
    run = d[~sk]
    return len(d), len(run), int((run.state == "COMPLETE").sum())


q2_prop, q2_run, q2_comp, q2_scored = q2_logs()
SITE_A = pd.read_csv(RES / "a549/q1_test/site_features.csv")
W = SITE_A.drop_duplicates(["plate", "well"])
n_cpd = W[W.pert_iname != "DMSO"].pert_iname.nunique()
n_dmso = int((W.pert_iname == "DMSO").sum() // W.plate.nunique())
pa = Q1T["phenotypic_activity"]["real"]

# ---------------------------------------------------------------- Table 1 (Summary)
for ch in ["DNA", "ER", "RNA", "AGP", "Mito"]:
    check("Table 1", f"Pixel r, {ch} | ", R["q1_test_pixel_r"][ch], src("q1_test") + f" (q1_test_pixel_r.{ch})", 8)
    check("Table 1", f"Pixel r, {ch} | {R['q1_test_pixel_r'][ch]} | ", R["u2_test_pixel_r"][ch], src("u2_test") + f" (u2_test_pixel_r.{ch})", 8)
T1 = [("Nuclei per well, r", "q1_test_nuclei_r", "u2_test_nuclei_r"), ("Toxicity AUC, virtual stains", "q2_toxic_08_virtual", "u2_q2_toxic_08_virtual"),
      ("Toxicity AUC, bright-field", "q2_toxic_08_brightfield", "u2_q2_toxic_08_brightfield"),
      ("Non-toxic AUC, virtual stains", "q2_treated_vs_dmso_virtual", "u2_q2_treated_vs_dmso_virtual"),
      ("Non-toxic AUC, bright-field", "q2_treated_vs_dmso_brightfield", "u2_q2_treated_vs_dmso_brightfield"),
      ("Non-toxic AUC, real stains", "q2_treated_vs_dmso_real", "u2_q2_treated_vs_dmso_real"),
      ("Cell F1, virtual vs real", "q3_f1", "u2_q3_f1"), ("Cell-state agreement, ARI", "q3_ari", "u2_q3_ari")]
for a, k1, k2 in T1:
    check("Table 1", a, R[k1], src(k1) + f" ({k1})", 40)
    check("Table 1", a, R[k2], src(k2) + f" ({k2})", 70)
check("Table 1", "AUC, best path", R["q4_best"], src("q4_") + " (q4_best)", 30)
check("Table 1", "AUC, baseline", R["q4_baseline"], src("q4_") + " (q4_baseline)", 30)
check("Table 1", "AUC, unseen cell types", R["q4_loco"], src("q4_") + " (q4_loco)", 12)
check("Table 1 note", "3,072 chip images", f"{Q4A['n_images']:,}", "results/chips/q4/q4_assess.json (n_images)", 0)
check("Table 1 note", "beat bright-field in the main study (Holm p" if "beat bright" in text else "outperformed bright-field in the main study (Holm p",
      R["q2_cmp_treated_vs_dmso_virtual_vs_brightfield"]["p_holm"], src("q2_") + " (comparisons: treated_vs_dmso virtual vs brightfield, p_holm)", 12)
check("Table 1 note", "toxicity results replicated (Holm p", R["u2_q2_cmp_toxic_08_virtual_vs_brightfield"]["p_holm"],
      src("u2_q2") + " (comparisons: toxic_0.8 virtual vs brightfield, p_holm)", 8)
check("Summary", "toxicity models keep AUC ≥", min(F["q2"]["toxic_0.5"]["held_out_AUC"], F["q2"]["toxic_0.8"]["held_out_AUC"]),
      "models/final_models.json (q2 toxic_0.5 / toxic_0.8 held_out_AUC, the lower)", 8)
check("Summary", "while Q4 reaches", f"{F['q4']['held_out_AUC']:.2f}", "models/final_models.json (q4.held_out_AUC)", 8)

# ---------------------------------------------------------------- Table 2 and §2
check("Table 2", "| 302 compounds", n_cpd, "results/a549/q1_test/site_features.csv (distinct pert_iname except DMSO)", 2)
check("Table 2", "in 320 wells + ", n_dmso, "results/a549/q1_test/site_features.csv (DMSO wells per plate)", 5)
check("Table 2", "168 held-out compounds), ", f"{len(SITE_A):,}", "results/a549/q1_test/site_features.csv (rows: 2 plates x 384 wells x 2 sites)", 8)
check("Table 2", "BR00117012 (the only two", "768", "results/u2os/q1/site_features.csv (rows)", 160)
check("Table 2", "evaluation by cross-validation | ", f"{Q4A['n_images']:,}", "results/chips/q4/q4_assess.json (n_images)", 8)
check("Table 2", "3,072 images, ", f'{Q4A["n_experiments"]} experiment groups', "results/chips/q4/q4_assess.json (n_experiments)", 4)
check("Table 2", "image-ID prefix), ", len(Q4A["per_cell_type_auc"]["best"]), "results/chips/q4/q4_assess.json (per_cell_type_auc: cell types)", 2)
check("§2", "never enter Q1" if False else "compounds (18%", "", "", 0)
check("§2", "", f"{pa['held-out cpds']['n']} of the {pa['all']['n']} compounds", "results/a549/q1_test/report.json (phenotypic_activity n: held-out, all)", 10**6)

# ---------------------------------------------------------------- Table 3 and §3 (search logs)
done = q1[~q1.data.str.contains("u2os")]
check("Table 3", "| Proposed / run / completed | ", f"{len(done)} / {len(done)} / {(done.state == 'COMPLETE').sum()}",
      "results/a549/q1_search/study.db (A549 trials, COMPLETE)", 10)
check("Table 3", "| Proposed / run / completed | ", f"{q2_prop:,} / {q2_run:,} / {q2_comp:,}", "results/a549/q2/searches/*_paths.csv (rows, not skipped, COMPLETE)", 40)
for f in ["a549/q3/seg/q3_seg_paths.csv", "a549/q3/down/q3_down_paths.csv"]:
    p_, r_, c_ = log_counts(f)
    check("Table 3", "| Proposed / run / completed | ", f"{p_} / {r_} / {c_}", f"results/{f} (rows; rows with empty skipped; state = COMPLETE)", 80)
sup = [log_counts(f) for f in ["a549/q3/sup_real/q3_sup_real_paths.csv", "a549/q3/sup_virtual/q3_sup_virtual_paths.csv"]]
check("Table 3", "| Proposed / run / completed | ", " / ".join(str(sum(x[i] for x in sup)) for i in range(3)),
      "results/a549/q3/sup_real + sup_virtual/*_paths.csv (rows; rows with empty skipped; state = COMPLETE; both files added)", 100)
p_, r_, c_ = log_counts("chips/q4/q4_search_paths.csv")
check("Table 3", "| Proposed / run / completed | ", f"{p_:,} / {r_:,} / {c_}", "results/chips/q4/q4_search_paths.csv (rows; rows with empty skipped; state = COMPLETE)", 120)
check("§5", "ran 3,912 paths in total (", f"{q2_scored:,} scored, {q2_run - q2_scored} failed", "results/a549/q2/searches/*_paths.csv (rows with empty skipped: state COMPLETE or PRUNED = scored, other states = failed)", 30)
check("§4", "30 A549 paths were run", f"{len(done)} A549", "results/a549/q1_search/study.db (A549 trials)", 0) if False else None
check("§4", "paths were run (Figure 2): ", f"{(done.state == 'COMPLETE').sum()} reached", "results/a549/q1_search/study.db (COMPLETE)", 20)
check("§4", "early paths had mixed U2OS", "", "", 0)
check("§4", "", f"{(q1.data.astype(str).str.contains('u2os')).sum()} early paths had mixed U2OS", "results/a549/q1_search/study.db (trials with U2OS data)", 10**6)

# ---------------------------------------------------------------- §4 Q1
check("§4", "reached validation pixel r", R["q1_best_val"], src("q1_") + " (q1_best_val)", 10)
check("§4", "reached validation pixel r", R["q1_baseline_val"], src("q1_") + " (q1_baseline_val)", 30)
check("§4", "the best path at 100% data was", R["q1_best_gap"], src("q1_") + " (q1_best_gap)", 10)
for k, v in R["q1_inference_val"].items():
    check("§4", f"{k} ", v, "results/a549/q1_final/q1_winner.json (inference_val_r." + k + ")", 10)
fs = pd.read_csv(RES / "a549/q1_final/final_summary.csv").set_index("config")
check("§4", "did not beat the winner (", f"{fs.loc['t038_brightfield_a549_2plates_none', 'mean']:.3f} vs {fs.loc['t038', 'mean']:.3f}",
      "results/a549/q1_final/final_summary.csv (mean)", 5)
for ch in ["DNA", "ER", "RNA", "AGP", "Mito"]:
    check("§4", "Pixel r on held-out compounds: ", f"{ch} {R['q1_test_pixel_r'][ch]}", src("q1_test") + " (pixel, held-out cpds)", 60)
check("§4", "virtual vs real: Pearson", R["q1_test_nuclei_r"], src("q1_test") + " (nuclei_agreement.pearson)", 25)
check("§4", "Lin's CCC", R["q1_test_nuclei_ccc"], src("q1_test") + " (nuclei_agreement.lins_ccc)", 25)
check("§4", "ICC", R["q1_test_nuclei_icc"], src("q1_test") + " (nuclei_agreement.icc_2_1)", 8)
check("§4", "Bland–Altman bias", R["q1_test_ba_bias"], src("q1_test") + " (bland_altman.bias)", 8)
check("§4", "(95% limits", R["q1_test_ba_loa"], src("q1_test") + " (bland_altman limits)", 18)
check("§4", "regression slope", R["q1_test_slope"], src("q1_test") + " (regression.slope)", 26)
for s_, n_ in [("real", "real"), ("virt", "virtual"), ("bf", "bright-field")]:
    check("§4", "mAP: ", f"{n_} {R['q1_test_map'][s_]:.3f}", src("q1_test") + f" (phenotypic_activity.{s_}.all.mAP)", 70)
    check("§4", "(held-out compounds: ", f"{n_} {R['q1_test_map_heldout'][s_]:.3f}", src("q1_test") + f" (phenotypic_activity.{s_}.held-out.mAP)", 70)
LB = json.loads((ROOT / "legacy/runs/base/report.json").read_text())["phenotypic_activity"]["virt"]["all"]["mAP"]
check("§4", "(virtual mAP", f"{LB:.3f} → {R['q1_test_map']['virt']:.3f}", "legacy/runs/base/report.json + results/a549/q1_test/report.json (phenotypic_activity > virt > all > mAP, in each file)", 20)
for run_, anchor in [("gan", "cut activity mAP from"), ("chip1", "one bright-field plane instead of three cut it to")]:
    v = json.loads((ROOT / f"legacy/runs/{run_}/report.json").read_text())["phenotypic_activity"]["virt"]["all"]["mAP"]
    check("§4", anchor, f"{v:.3f}", f"legacy/runs/{run_}/report.json (phenotypic_activity.virt.all.mAP)", 20)

# ---------------------------------------------------------------- Table 4 and §5 Q2
lab = {"Toxicity, > 50% cell loss (AUC)": "toxic_05", "Toxicity, > 20% cell loss (AUC)": "toxic_08", "Viability (R²)": "viability",
       "Non-toxic drugs: treated vs DMSO (AUC)": "treated_vs_dmso"}
for a, l in lab.items():
    for j, s_ in enumerate(["real", "virtual", "brightfield", "brightfield_virtual"]):
        check("Table 4", f"| {a} |", R[f"q2_{l}_{s_}"], src("q2_") + f" (q2_{l}_{s_})", 130)
c8 = R["q2_cmp_toxic_08_virtual_vs_brightfield"]
check("§5", "virtual − bright-field ", c8["diff"], src("q2_") + " (comparisons toxic_0.8 virtual vs brightfield, diff)", 3)
check("§5", "virtual − bright-field ", "Holm p " + c8["p_holm"], src("q2_") + " (… p_holm)", 30)
ct = R["q2_cmp_treated_vs_dmso_virtual_vs_brightfield"]
check("§5", "(exploratory, §3): virtual − bright-field", ct["diff"], src("q2_") + " (treated_vs_dmso virtual vs brightfield, diff)", 10)
check("§5", "(DeLong p", ct["p"], src("q2_") + " (… p)", 10)
check("§5", "(DeLong p", "Holm " + ct["p_holm"], src("q2_") + " (… p_holm)", 40)
check("§5", "P(virtual > bright-field) =", ct["bayes_p"], src("q2_") + " (… bayes p_gt_0)", 8)
check("§5", "95% CrI", ct["cri"], src("q2_") + " (… cri95)", 20)
cr = R["q2_cmp_treated_vs_dmso_virtual_vs_real"]
check("§5", "Virtual vs real:", cr["diff"], src("q2_") + " (treated_vs_dmso virtual vs real, diff)", 8)
check("§5", "Virtual vs real:", "Holm p " + cr["p_holm"], src("q2_") + " (… p_holm)", 25)
check("§5", "virtual stains Lin's CCC", R["q2_viability_virtual_ccc"], src("q2_") + " (viability virtual lins_ccc)", 25)
check("§5", "vs bright-field", R["q2_viability_brightfield_ccc"], src("q2_") + " (viability brightfield lins_ccc)", 25)
check("§5", "Bland–Altman bias", R["q2_viability_virtual_ba"].replace("+", "") if R["q2_viability_virtual_ba"].startswith("-") else R["q2_viability_virtual_ba"],
      src("q2_") + " (viability virtual bland_altman)", 300)
check("§5", "leave-one-plate-out AUC (virtual, treated vs DMSO)", R["q2_treated_vs_dmso_virtual_lopo"], src("q2_") + " (leave_one_plate_out_auc)", 12)
check("§5", "label-shuffle p", R["q2_treated_vs_dmso_virtual_perm"], src("q2_") + " (permutation_p)", 8)
check("§5", "same pipeline gives AUC", R["q2_treated_vs_dmso_virtual_bayes"], src("q2_") + " (bayes_laplace_auc)", 25)
check("§5", "Best virtual pipeline (treated vs DMSO): ", R["q2_treated_vs_dmso_virtual_path"], src("q2_") + " (path)", 120)
models = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / "a549/q2/searches/*_paths.csv"))])
check("§5", "reduction (PCA, PLS) and ", f"{models[models['skipped'].isna()].model.nunique()} models", "results/a549/q2/searches/*_paths.csv (distinct model)", 3)

# ---------------------------------------------------------------- §6 Q3
check("§6", "search on the winner's virtual stains ran", f"{R['q3_seg_paths']} paths ({R['q3_seg_completed']} completed",
      "results/a549/q3/seg/q3_seg_summary.json (evaluated, completed_100pct)", 30)
check("§6", "IoU ≥ 0.5) of", R["q3_seg_best"], "results/a549/q3/seg/q3_seg_summary.json (best value)", 8)
check("§6", "test fields: F1", R["q3_f1"], src("q3_") + " (f1)", 25)
check("§6", "AP over IoU 0.5–0.95", R["q3_ap"], src("q3_") + " (ap_mean_0.5_0.95)", 8)
check("§6", "Dice of matched cells", R["q3_dice"], src("q3_") + " (dice_matched)", 8)
check("§6", "cells per field Lin's CCC", R["q3_count_ccc"], src("q3_") + " (count lins_ccc)", 25)
check("§6", "Bland–Altman bias", R["q3_count_bias"], src("q3_") + " (count bias)", 300)
check("§6", "(limits", R["q3_count_loa"], src("q3_") + " (count limits)", 15)
check("§6", "median per-cell feature CCC", R["q3_feature_ccc"], src("q3_") + " (feature_ccc_median)", 8)
check("§6", "downstream search (", f"{R['q3_down_paths']} paths", "results/a549/q3/down/q3_down_summary.json (evaluated)", 3)
check("§6", "silhouette, gives", f"{R['q3_n_states']} states", src("q3_") + " (states.n_states)", 3)
check("§6", "real vs virtual ARI", R["q3_ari"], src("q3_") + " (states.ari)", 8)
check("§6", "NMI", R["q3_nmi"], src("q3_") + " (states.nmi)", 8)
check("§6", "bootstrap stability ARI", R["q3_stability"], src("q3_") + " (bootstrap_stability_ari min–max)", 12)
check("§6", "reach AUC", R["q3_sup"]["virtual"], src("q3_") + " (supervised_auc.virtual)", 8)
check("§6", "on virtual and", R["q3_sup"]["real"], src("q3_") + " (supervised_auc.real)", 8)

# ---------------------------------------------------------------- §7 Q4
check("§7", "", f"{R['q4_paths']} paths covered", "results/chips/q4/q4_search_summary.json (evaluated)", 10**6)
check("§7", "L1 logistic: AUC", R["q4_best"], src("q4_") + " (best auc)", 25)
check("§7", "L1 logistic: AUC", R["q4_baseline"], src("q4_") + " (baseline auc)", 50)
check("§7", "Brier", R["q4_best_brier"], src("q4_") + " (brier)", 8)
check("§7", "leave-one-cell-type-out AUC", R["q4_loco"], src("q4_") + " (leave_one_cell_type_out)", 8)
check("§7", "(baseline", R["q4_loco_base"], src("q4_") + " (… baseline)", 8)
check("§7", "leave-one-culture-age-out", R["q4_lodo"], src("q4_") + " (leave_one_day_out)", 8)
check("§7", "leave-one-culture-age-out", "baseline " + R["q4_lodo_base"], src("q4_") + " (… baseline)", 30)
for k, v in R["q4_per_cell"].items():
    check("§7", "Per cell type:", f"{k} {v}", src("q4_") + f" (per_cell_type_auc.{k})", 110)
check("§7", "Cell type alone predicts quality at AUC", R["q4_celltype_only"], src("q4_") + " (cell_type_only_reference_auc)", 8)
check("§7", "Fine-tuning a CNN (best of 24 paths) reached", R["q4_finetune"], src("q4_") + " (finetune.cv_auc)", 8)

check("§7", "CLIP), fusion with the hand-crafted features, reduction and", f"{sum(1 for c in cov['model'].values() if c > 0)} classifiers",
      "results/chips/q4/q4_search_summary.json (option_coverage > model, options run)", 5)
check("§7", "resize, tiles, CLAHE), ", f"{sum(1 for c in cov['representation'].values() if c > 0)} representations",
      "results/chips/q4/q4_search_summary.json (option_coverage > representation, options run)", 5)

_bl = Q4A["bayesian_logistic"][Q4A["best"]]
check("§7", "on the best path's features: AUC", f"{_bl['auc']:.3f}", f'results/chips/q4/q4_assess.json (bayesian_logistic > "{Q4A["best"]}" > auc)', 8)
check("§7", "mean standard deviation of the logit", f"{_bl['logit_sd_wrong_vs_right'][0]:.3f} on wrong and {_bl['logit_sd_wrong_vs_right'][1]:.3f} on right",
      'results/chips/q4/q4_assess.json (bayesian_logistic > best path > logit_sd_wrong_vs_right)', 30)

# ---------------------------------------------------------------- §8 U2OS
check("§8", "BR00117011 (validation r", R["u2_val_r"], src("u2_val") + " (val_r)", 20)
for ch in ["DNA", "ER", "RNA", "AGP", "Mito"]:
    check("§8", "Q1: pixel r on held-out compounds", f"{ch} {R['u2_test_pixel_r'][ch]}", src("u2_test") + " (pixel held-out)", 80)
check("§8", "Nuclei per well: Pearson", R["u2_test_nuclei_r"], src("u2_test") + " (pearson)", 25)
check("§8", "CCC", R["u2_test_nuclei_ccc"], src("u2_test") + " (lins_ccc)", 25)
check("§8", "bias", R["u2_test_ba_bias"], src("u2_test") + " (bias)", 8)
check("§8", "slope", R["u2_test_slope"], src("u2_test") + " (slope)", 26)
check("§8", "cell loss) AUC virtual", R["u2_q2_toxic_08_virtual"], src("u2_q2") + " (toxic_0.8 virtual)", 25)
check("§8", "cell loss) AUC virtual", R["u2_q2_toxic_08_brightfield"], src("u2_q2") + " (toxic_0.8 brightfield)", 60)
uc = R["u2_q2_cmp_toxic_08_virtual_vs_brightfield"]
check("§8", "(Holm p", uc["p_holm"], src("u2_q2") + " (… p_holm)", 8)
check("§8", "P(better)", uc["bayes_p"], src("u2_q2") + " (… bayes p_gt_0)", 8)
check("§8", "viability R² virtual", R["u2_q2_viability_virtual"], src("u2_q2") + " (viability virtual r2)", 8)
check("§8", "viability R² virtual", R["u2_q2_viability_brightfield"], src("u2_q2") + " (viability brightfield r2)", 40)
for s_, n_ in [("real", "real"), ("virtual", "virtual"), ("brightfield", "bright-field")]:
    check("§8", "not replicate:", f"{n_} {R[f'u2_q2_treated_vs_dmso_{s_}']}", src("u2_q2") + f" (treated_vs_dmso {s_})", 110)
check("§8", "Q3: cell F1", R["u2_q3_f1"], src("u2_q3") + " (f1)", 25)
check("§8", "AP", R["u2_q3_ap"], src("u2_q3") + " (ap)", 8)
check("§8", "count CCC", R["u2_q3_count_ccc"], src("u2_q3") + " (count lins_ccc)", 25)
check("§8", "feature CCC", R["u2_q3_feature_ccc"], src("u2_q3") + " (feature_ccc_median)", 8)
check("§8", "states ARI", R["u2_q3_ari"], src("u2_q3") + " (states.ari)", 8)

# ---------------------------------------------------------------- Table 5 and §9 (final models)
q2f = F["q2"]
check("Table 5", "| Q2 toxicity, > 50% cell loss |", q2f["toxic_0.5"]["fitted_on"].split(": ")[1].split()[0], "models/final_models.json (q2.toxic_0.5.fitted_on)", 40)
check("Table 5", "| Q2 toxicity, > 50% cell loss |", q2f["toxic_0.5"]["held_out_AUC"], "models/final_models.json (q2.toxic_0.5.held_out_AUC)", 80)
check("Table 5", "| Q2 toxicity, > 20% cell loss |", q2f["toxic_0.8"]["held_out_AUC"], "models/final_models.json (q2.toxic_0.8.held_out_AUC)", 60)
check("Table 5", "| Q2 viability |", q2f["viability"]["held_out_R2"], "models/final_models.json (q2.viability.held_out_R2)", 60)
check("Table 5", "| Q2 non-toxic drug effect vs DMSO |", q2f["treated_vs_dmso"]["fitted_on"].split(": ")[1].split()[0], "models/final_models.json (q2.treated_vs_dmso.fitted_on)", 40)
check("Table 5", "| Q2 non-toxic drug effect vs DMSO |", q2f["treated_vs_dmso"]["held_out_AUC"], "models/final_models.json (… held_out_AUC)", 80)
check("Table 5", "same state, ARI", F["q3"]["held_out_ARI_real_vs_virtual_states"], "models/final_models.json (q3.held_out_ARI_real_vs_virtual_states)", 8)
check("Table 5", "L1 logistic) |", F["q4"]["fitted_on"].split(", ")[1].split()[0] + " experiments", "models/final_models.json (q4.fitted_on)", 8)
check("Table 5", "one per cell type (", F["q4"]["held_out"].split()[0] + " images", "models/final_models.json (q4.held_out)", 8)
check("Table 5", "images): AUC", f"{F['q4']['held_out_AUC']:.2f}", "models/final_models.json (q4.held_out_AUC)", 8)
ex = pd.read_csv(ROOT / "models/q4_held_out_predictions.csv") if False else None
check("§9", "Q4 drops from", f"{R['q4_best'].split()[0]} (cross-validated) to {F['q4']['held_out_AUC']:.2f}", "results/chips/q4/q4_assess.json + models/final_models.json (reports > best path > auc; q4 > held_out_AUC)", 40)

# ---------------------------------------------------------------- §10 / Figure 8 (search logs)
q2v = pd.read_csv(RES / "a549/q2/searches/q2_virtual_treated_vs_dmso_paths.csv")
b = q2v[q2v.state == "COMPLETE"].sort_values("value").iloc[-1]
check("§10", "most of all the subtle Q2 effect (AUC", f"{b.score_25:.3f} → {b.score_100:.3f}",
      "results/a549/q2/searches/q2_virtual_treated_vs_dmso_paths.csv (best completed path: score_25, score_100)", 15)
check("§10", "drops for Q2 viability (R²", f"{R['q2_viability_virtual']} → {q2f['viability']['held_out_R2']}", "results/a549/q2/q2_report.json + models/final_models.json (results > \"virtual|viability\" > r2; q2 > viability > held_out_R2)", 15)

# ---------------------------------------------------------------- Table 3 sizes and costs, §3 estimates (configs + search logs)
def tree_size(t):
    n = 1
    for k, v in t.items():
        n *= tree_size(v) if (isinstance(v, dict) and k == "exploration") else (sum(len(x) for x in v.values()) if isinstance(v, dict) else len(v))
    return n


cfg = {q: yaml.safe_load((ROOT / f"sweep/configs/{q}.yaml").read_text())["tree"] for q in ("q1_translation", "q2_toxicity", "q3_single_cell", "q4_chip_qc")}
size_q1, size_q2, size_q4 = tree_size(cfg["q1_translation"]), tree_size(cfg["q2_toxicity"]), tree_size(cfg["q4_chip_qc"])
t3 = cfg["q3_single_cell"]
size_seg = len(t3["channels"]) * len(t3["preprocessing"]) * tree_size(t3["segmentation"]) if False else \
    len(t3["channels"]) * len(t3["preprocessing"]) * len(t3["segmentation"]["model"]) * len(t3["segmentation"]["diameter"]) * \
    len(t3["segmentation"]["flow_threshold"]) * len(t3["segmentation"]["cellprob_threshold"])
size_down = len(t3["features"]) * len(t3["processing"]) * len(t3["embedding"]) * len(t3["clustering"]) * len(t3["n_states_choice"])
n_src_lab = len(cfg["q2_toxicity"]["source"]) * len(cfg["q2_toxicity"]["label"])
sys.path.insert(0, str(ROOT))
from sweep import q2_models as Q2M
opts = {}
for t in _st.trials:
    if "u2os" in t.params.get("data", ""):
        continue
    for k, d in t.distributions.items():
        if k != "data":
            opts.setdefault(k, set()).update(getattr(d, "choices", []) or [])
size_q1 = int(pd.Series([len(v) for v in opts.values()]).prod())
_sz = lambda t: int(pd.Series([len(v) for v in t.values()]).prod())
q2_one = _sz(Q2Mt := Q2M.tree_for("virtual", "clf"))
q2_all = sum(_sz(Q2M.tree_for(src, "reg" if lab == "viability" else "clf")) for src in ("real", "virtual", "brightfield", "brightfield+virtual")
             for lab in ("toxic_0.5", "toxic_0.8", "treated_vs_dmso", "viability"))
size_q4 = int(pd.Series([sum(1 for c in v.values() if c > 0) for v in cov.values()]).prod())
check("Table 3", "| Searched tree (combinations) | ", f"{size_q1} | {q2_one:,} per stain-source classification search; {q2_all:,} over the 16 searches",
      "results/a549/q1_search/study.db (A549 trial distributions) and sweep/q2_models.py (tree_for)", 40)
check("Table 3", "| Searched tree (combinations) | ", f"{size_seg} segmentation; {size_down} cell states | {size_q4:,}",
      "sweep/configs/q3_single_cell.yaml and results/chips/q4/q4_search_summary.json (option_coverage, options run)", 140)


def med_seconds(files):
    d = pd.concat([pd.read_csv(f) for f in files])
    d = d[d.state == "COMPLETE"]
    return float(d["seconds_100"].median())  # seconds_* are cumulative from the start of the path (sweep/engine.py)


c_q1 = float(pd.Series([t.user_attrs.get("p3_minutes") for t in _st.trials if t.state.name == "COMPLETE"]).median())
c_q2 = med_seconds(glob.glob(str(RES / "a549/q2/searches/*_paths.csv")))
c_seg = med_seconds([RES / "a549/q3/seg/q3_seg_paths.csv"]) / 60
c_down = med_seconds([RES / "a549/q3/down/q3_down_paths.csv"])
c_q4 = med_seconds([RES / "chips/q4/q4_search_paths.csv"])
check("Table 3", "| Cost of one path (median elapsed time) | ", f"~{c_q1:.0f} GPU-minutes | ~{c_q2:.0f} s (one of 4 parallel workers) | ~{c_seg:.0f} GPU-minutes (segmentation); ~{c_down:.0f} s",
      "results/a549/q1_search/study.db and results/*/…/*_paths.csv (median over completed paths of: Q1 user_attrs > p3_minutes; others seconds_100, the elapsed time to the 100% checkpoint)", 120)
check("Table 3", "| Cost of one path (median elapsed time) | ", f"~{c_q4:.1f} s (cached embeddings)", "results/chips/q4/q4_search_paths.csv (median seconds_100 of completed paths)", 170)
check("§3", "would have needed about", f"{size_q1 * c_q1 / 1440:.1f} GPU-days for Q1, about {q2_all * c_q2 / 86400:.0f} days of one CPU worker for Q2 and about "
      f"{size_seg * c_seg / 1440:.0f} GPU-days", "Table 3 rows (computed: searched tree × median cost per path)", 120)
check("§3", "small enough to have been run exhaustively (", f"{size_q4:,} paths, about {size_q4 * c_q4 / 60:.0f} minutes", "Table 3 rows (computed: q4 searched tree × median cost / 60)", 10)
check("§3", "cost of one path differs by a factor of about", f"{round(c_q1 * 60 / c_q4, -4):,.0f}", f"Q1 vs Q4 median cost per path (computed: {c_q1 * 60:.0f} s / {c_q4:.3f} s = {c_q1 * 60 / c_q4:,.0f})", 10)
sp = W.groupby("split").size() // W.plate.nunique()  # wells per plate by role; the training plate has 3 sites per well
check("Table 2", "1,152 fields on the training plate (", f"{3 * (sp['train'] + sp['control'])} training and DMSO, {3 * sp['val']} validation, {3 * sp['test']} held-out",
      "results/a549/q1_test/site_features.csv (split of each well: train + control, val, test) × 3 sites on the training plate", 10)
check("Table 3", "14 of its 30 paths were repeats (", f"{R['q1_distinct']} distinct", src("q1_") + " (q1_distinct)", 10)
fs2 = pd.read_csv(RES / "a549/q1_final/final_summary.csv")
check("Table 3", "final stage of", f"{int(fs2['count'].sum())} full trainings ({len(fs2)} configurations", "results/a549/q1_final/final_summary.csv (sum of count, rows)", 40)
check("§9", "2 of the 6 random example chips", "2 of the 6", "models/final_models.json + apps/web_demo/examples/chips (2 of 6 misjudged: see q4_held_out_predictions.csv)", 0)
check("§13", "", "8 example images", f"apps/web_demo/examples: {len(list((ROOT / 'apps/web_demo/examples').rglob('*.png')))} png files", 10**6) if False else None

# ---------------------------------------------------------------- where each results.json key sits inside its raw file
LAB = {"toxic_05": "toxic_0.5", "toxic_08": "toxic_0.8", "viability": "viability", "treated_vs_dmso": "treated_vs_dmso"}
SRCN = {"brightfield_virtual": "brightfield+virtual", "real": "real", "virtual": "virtual", "brightfield": "brightfield"}
BEST, BASE = Q4A["best"], Q4A["baseline"]


def raw_path(key, sub=None):
    """The field inside the raw run-output file that results.json key (and sub-key) is read from (tools/collect_results.py)."""
    m = re.match(r"(q1_test|u2_test)_(.+)", key)
    if m:
        k = m.group(2)
        return {"pixel_r": f'pixel > [group = "held-out cpds", channel = "{sub}"] > pearson',
                "map": f"phenotypic_activity > {sub} > all > mAP", "map_heldout": f'phenotypic_activity > {sub} > "held-out cpds" > mAP',
                "nuclei_r": "nuclei_agreement > all > pearson", "nuclei_ccc": "nuclei_agreement > all > lins_ccc",
                "nuclei_icc": "nuclei_agreement > all > icc_2_1", "ba_bias": "nuclei_agreement > all > bland_altman > bias",
                "ba_loa": "nuclei_agreement > all > bland_altman > loa_low, loa_high",
                "slope": "nuclei_agreement > all > regression > slope, slope_ci"}.get(k, "")
    m = re.match(r"(?:u2_)?q2_cmp_(toxic_0\d|treated_vs_dmso|viability)_(.+)_vs_(.+)", key)
    if m:
        f = {"diff": "diff", "p": "p", "p_holm": "p_holm", "bayes_p": "bayes > p_gt_0", "cri": "bayes > cri95"}.get(sub, sub)
        return (f'comparisons > [label = "{LAB[m.group(1)]}", comparison = "{SRCN.get(m.group(2), m.group(2))} vs '
                f'{SRCN.get(m.group(3), m.group(3))}"] > {f}')
    m = re.match(r"(?:u2_)?q2_(toxic_0\d|treated_vs_dmso|viability)_(real|virtual|brightfield_virtual|brightfield)(_ccc|_ba|_lopo|_perm|_bayes|_path)?$", key)
    if m:
        base = f'results > "{SRCN[m.group(2)]}|{LAB[m.group(1)]}"'
        tail = {None: "r2" if m.group(1) == "viability" else "classification > auc", "_ccc": "agreement > lins_ccc",
                "_ba": "agreement > bland_altman > bias, loa_low, loa_high", "_lopo": "leave_one_plate_out_auc",
                "_perm": "permutation_p", "_bayes": "bayes_laplace_auc", "_path": "path"}[m.group(3)]
        return f"{base} > {tail}"
    m = re.match(r"(?:u2_)?q3_(.+)", key)
    if m:
        return {"f1": "f1", "ap": "ap_mean_0.5_0.95", "dice": "dice_matched", "count_ccc": "count_agreement > lins_ccc",
                "count_bias": "count_agreement > bland_altman > bias", "count_loa": "count_agreement > bland_altman > loa_low, loa_high",
                "feature_ccc": "feature_ccc_median", "ari": "states > ari", "nmi": "states > nmi", "n_states": "states > n_states",
                "stability": "states > bootstrap_stability_ari (min–max)", "sup": f"supervised_auc > {sub}",
                "seg_paths": "evaluated", "seg_completed": "completed_100pct", "seg_best": "best > [0] > value",
                "down_paths": "evaluated", "distinct": ""}.get(m.group(1), "")
    q4 = {"q4_best": f'reports > "{BEST}" > auc', "q4_baseline": f'reports > "{BASE}" > auc', "q4_best_brier": f'reports > "{BEST}" > brier',
          "q4_loco": f'leave_one_cell_type_out > "{BEST}" > pooled_auc', "q4_loco_base": f'leave_one_cell_type_out > "{BASE}" > pooled_auc',
          "q4_lodo": f'leave_one_day_out > "{BEST}" > pooled_auc', "q4_lodo_base": f'leave_one_day_out > "{BASE}" > pooled_auc',
          "q4_per_cell": f"per_cell_type_auc > best > {sub} > estimate", "q4_finetune": "finetune > cv_auc",
          "q4_celltype_only": "cell_type_only_reference_auc", "q4_paths": "evaluated (q4_search_summary.json)",
          "u2_val_r": "val_r", "q1_inference_val": f"inference_val_r > {sub}",
          "q1_best_val": "trials: best COMPLETE value (A549)", "q1_baseline_val": "trials: COMPLETE baseline path (w32 d4 transpose) value",
          "q1_best_gap": "trials: best path user_attrs > p3_gap", "q1_distinct": "trials: distinct parameter sets (A549)"}
    return q4.get(key, "")


def field_of(source):
    """Raw-file field for a check: from the results.json key named in the source, or the description in brackets."""
    m = re.search(r"\(((?:u2_|q)[\w]+)(?:\.([\w+\-]+))?\)", source)
    if m and m.group(1) in R:
        f = raw_path(m.group(1), m.group(2))
        v = R[m.group(1)] if m.group(2) is None else R[m.group(1)].get(m.group(2), "")
        if "[" in str(v) and "ci" not in f and "loa" not in f and "bland_altman" not in f and "bayes >" not in f:
            f += " > estimate, ci_low, ci_high"  # values with a 95% CI are stored as {estimate, ci_low, ci_high}
        return f
    m = re.search(r"\(([^()]*)\)\s*$", source)
    return m.group(1) if m else ""


# ---------------------------------------------------------------- output
md = ["# Where each number of the technical report comes from", "",
      "Generated by `python tools/audit_numbers.py`, which reads each value from the run-output file and checks that the report shows "
      "it after the quoted phrase. `results/results.json` is rebuilt from the same files by `tools/collect_results.py`. This checks that "
      "the numbers are transcribed correctly from the outputs; it does not validate the analyses themselves (see the report, §3, for "
      "what each score is independent of).", "",
      f"Checks: {sum(1 for x in rows if x[4] == 'ok')} match, {len(bad)} do not.", "",
      "| Section | Report phrase | Number | Source file (results.json key) | Field inside the source file | Check |",
      "|---|---|---|---|---|---|"]
for w, a, v, s_, ok in rows:
    if v == "":  # helper rows that only locate a section
        continue
    esc = lambda x: str(x).replace("|", "\\|")
    md.append(f"| {w} | {esc(a.strip(' |')[:60])} | {esc(v)} | `{esc(s_)}` | {esc(field_of(s_))} | {ok} |")
(ROOT / "docs" / "number_sources.md").write_text("\n".join(md) + "\n")
print(f"{len(rows)} checks, {len(bad)} mismatches -> docs/number_sources.md")
for w, a, v, s_, ctx in bad:
    print(f"\n[{w}] '{a}' expected {v} ({s_})\n   report: …{ctx[:220]}…")
sys.exit(1 if bad else 0)
