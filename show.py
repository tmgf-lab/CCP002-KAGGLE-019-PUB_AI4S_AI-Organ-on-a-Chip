"""Short, readable summaries of result files (used in the demo video; works on any run folder).

    python show.py q1search <VCP_RUNS>             # every path of a Q1 search: checkpoints and outcome
    python show.py q1test <evaluate --out folder>  # pixel r per stain (held-out compounds), nuclei agreement
    python show.py search <*_summary.json>         # any engine search (Q2/Q3/Q4): paths, exclusions, best path
    python show.py q3 <Q3_OUT>                     # real vs virtual cells: F1, AP, counts, features, states
"""
import json
import sys
from pathlib import Path


def q1search(runs):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    st = optuna.load_study(study_name="q1", storage=f"sqlite:///{Path(runs) / 'q1_translation' / 'study.db'}")
    print(f"{'path':<62} {'25%':>6} {'50%':>6} {'100%':>6}  outcome")
    for t in st.trials:
        p = t.params
        name = f"{p.get('processing')} > U-Net w{p.get('width')} {p.get('upsampling')}" + (" +attention" if p.get("attention") else "")
        v = [f"{t.intermediate_values[k]:.3f}" if k in t.intermediate_values else "" for k in (1, 2, 3)]
        out = {"COMPLETE": "completed", "PRUNED": "stopped early"}.get(t.state.name, t.state.name.lower())
        if t.user_attrs.get("skipped"):
            out = "skipped (repeat of an evaluated path)"
        print(f"{name:<62} {v[0]:>6} {v[1]:>6} {v[2]:>6}  {out}")
    done = [t for t in st.trials if t.state.name == "COMPLETE"]
    if done:
        best = max(done, key=lambda t: t.value)
        print(f"\nbest path: trial {best.number}, validation r {best.value:.4f}")
    else:
        print("\nno path completed")


def q1test(d):
    r = json.loads((Path(d) / "report.json").read_text())
    pix = {x["channel"]: x["pearson"] for x in r["pixel"] if x["group"] == "held-out cpds"}
    print("pixel r, virtual vs real, held-out compounds:  " + "  ".join(f"{c} {pix[c]:.3f}" for c in ("DNA", "ER", "RNA", "AGP", "Mito")))
    a = r["nuclei_agreement"]["all"]
    print(f"nuclei per well, virtual vs real:  r {a['pearson']['estimate']:.3f}   Lin's CCC {a['lins_ccc']['estimate']:.3f}   "
          f"Bland-Altman bias {a['bland_altman']['bias']:+.2f} nuclei")


def search(f):
    s = json.loads(Path(f).read_text())
    print(f"{s['name']}: {s['evaluated']} paths evaluated, {s['completed_100pct']} reached 100% of the data")
    print("options excluded (4 bad results):", ", ".join(f"{e['node']}={e['option']}" for e in s["excluded_options"]) or "none")
    b = s["best"][0]
    keys = [k for k in s["option_coverage"]]
    print("best path:", " > ".join(str(b[k]) for k in keys), f"  score {b['value']:.3f}")


def q3(d):
    r = json.loads((Path(d) / "q3_report.json").read_text())
    n = len(json.loads((Path(d) / "seg_fields.json").read_text())["per_field"])
    st = r.get("states", {})
    print(f"{n} fields, the same segmentation on real and virtual stains:")
    print(f"  cells found (F1, IoU >= 0.5) {r['f1']['estimate']:.3f}   mask AP {r['ap_mean_0.5_0.95']:.3f}   "
          f"cells per field CCC {r['count_agreement']['lins_ccc']['estimate']:.3f}")
    print(f"  per-cell features (median CCC) {r['feature_ccc_median']:.3f}   cell states real vs virtual ARI {st.get('ari', float('nan')):.3f}")


if __name__ == "__main__":
    {"q1search": q1search, "q1test": q1test, "search": search, "q3": q3}[sys.argv[1]](sys.argv[2])
