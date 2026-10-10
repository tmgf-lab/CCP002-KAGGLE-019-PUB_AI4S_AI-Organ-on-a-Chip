"""Final report figures, drawn from the saved results of the four searches (no model is re-run here).

    python tools/report_figures.py      # writes figures/fig_*.png for every question whose results exist
    VCP_RAW_RUNS=<raw runs folder> VCP_DATA=<plates folder> python tools/report_figures.py   # also the image examples
"""
import json
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os
ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
RAW = Path(os.environ.get("VCP_RAW_RUNS", ROOT / "runs"))  # raw outputs (virtual-stain caches), only for the image examples
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
INK, INK2, GRID, SURF, GREY = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb", "#a9a8a2"
COL = {"real": "#2a78d6", "virtual": "#eb6834", "brightfield": "#1baf7a", "brightfield+virtual": "#4a3aa7"}
LBL = {"real": "Real stains", "virtual": "Virtual stains", "brightfield": "Bright-field only",
       "brightfield+virtual": "Bright-field + virtual"}
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "font.size": 10,
                     "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": GRID, "axes.axisbelow": True, "axes.titlesize": 11, "axes.titleweight": "bold",
                     "axes.titlelocation": "left"})


def note(fig, text):
    fig.text(0.01, 0.01, text, fontsize=8, color=INK2, va="bottom")


# ================================================================================================ Q1
def q1_label(p):
    if p.get("model") == "unet_resnet34_pretrained":
        arch = "ResNet34 encoder (pretrained)"
    else:
        arch = f"U-Net w{p['width']} d{p['depth']}, {p['upsampling']}"
    extra = [x for x in (p.get("processing", "") if p.get("processing") != "plate_scaling" else "",
                         "+U2OS" if "u2os" in p.get("data", "") else "", "attention" if p.get("attention") else "") if x]
    return arch + (" (" + ", ".join(extra) + ")" if extra else "")


def fig_q1_search():
    db = RES / "a549" / "q1_search" / "study.db"
    if not db.exists():
        return
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.load_study(study_name="q1", storage=f"sqlite:///{db}")
    trials = [t for t in study.trials if t.intermediate_values and "u2os" not in t.params.get("data", "")]  # A549 study only
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]
    frac = {1: 25, 2: 50, 3: 100}
    lo = 0.78  # bottom of the axis; a score below it is drawn on the edge as an off-scale marker
    for t in trials:
        done = t.state.name == "COMPLETE"
        steps = sorted(t.intermediate_values)
        x = [frac[s] for s in steps]
        y = [max(t.intermediate_values[s], lo + 0.002) for s in steps]
        c = COL["virtual"] if done else GREY
        ax.plot(x, y, color=c, alpha=0.9 if done else 0.6, lw=1.4 if done else 1, marker="o", ms=3)
        for s in steps:
            if t.intermediate_values[s] < lo:
                ax.plot(frac[s], lo + 0.002, marker="v", ms=8, color=c, clip_on=False)
                ax.annotate(f"r {t.intermediate_values[s]:.2f} (off scale)", (frac[s], lo + 0.002), xytext=(6, 4),
                            textcoords="offset points", fontsize=7.5, color=INK2)
        if min(t.intermediate_values.values()) < lo and 3 in t.intermediate_values:
            ax.annotate("pretrained ResNet34 encoder", (100, t.intermediate_values[3]), xytext=(-6, -14),
                        textcoords="offset points", ha="right", fontsize=7.5, color=INK2)
    ax.set(xticks=[25, 50, 100], xlabel="training data used (%)", ylabel="validation pixel r", title="Q1 search: every path, every checkpoint")
    ax.set_ylim(lo, 0.965)
    ax.plot([], [], color=COL["virtual"], label="completed (100%)")
    ax.plot([], [], color=GREY, label="stopped early (worse half)")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    ax = axes[1]
    best = {}
    for t in trials:
        if t.state.name == "COMPLETE":
            k = q1_label(t.params)
            best[k] = max(best.get(k, -1), t.value)
    best = pd.Series(best).sort_values().tail(6)
    # all of these completed: the best path dark, the runners-up light
    ax.barh(range(len(best)), best.values, color=[COL["virtual"] if i == len(best) - 1 else "#f4b896" for i in range(len(best))],
            edgecolor=SURF, linewidth=2)
    for i, v in enumerate(best.values):
        ax.text(v + 0.0003, i, f"{v:.4f}", va="center", fontsize=8.5)
    ax.set_yticks(range(len(best)), best.index, fontsize=8.5)
    ax.set_xlim(max(0.94, best.min() - 0.003), best.max() + 0.004)
    ax.set(title="Best distinct paths (validation, 100% data)", xlabel="validation pixel r")
    ax.grid(axis="y", visible=False)
    n_stop = sum(t.state.name == "PRUNED" for t in trials)
    n_done = sum(t.state.name == "COMPLETE" for t in trials)
    n_mixed = sum(1 for t in study.trials if "u2os" in t.params.get("data", ""))
    n_unscored = sum(1 for t in study.trials if not t.intermediate_values and "u2os" not in t.params.get("data", ""))
    note(fig, f"{len(trials) + n_unscored} A549 paths run, {len(trials)} with a score (shown): {n_done} completed, {n_stop} stopped early, "
              f"{len(trials) - n_done - n_stop} cut by the time limit (TPE + successive halving, 2 T4 GPUs; "
              f"validation fields only).\nA path is stopped only against paths already scored at the same checkpoint, so the second path (pretrained encoder) reached 100% without competition.\n{n_mixed} early paths that mixed U2OS into training are excluded: datasets are separate studies. Right: best path dark, runners-up light.")
    fig.tight_layout(rect=(0, 0.11, 1, 0.93))
    fig.savefig(OUT / "fig_q1_search.png", dpi=170)
    plt.close(fig)


# ================================================================================================ Q2
def fig_q2():
    f = RES / "a549" / "q2" / "q2_report.json"  # rerun on the Q1 winner's virtual stains
    if not f.exists():
        return
    rep = json.loads(f.read_text())["results"]
    labels = [("toxic_0.5", "Kills > 50% of cells"), ("toxic_0.8", "Kills > 20% of cells"),
              ("treated_vs_dmso", "Non-toxic: treated vs DMSO"), ("viability", "Viability (R²)")]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.6), sharey=False)
    for ax, (lab, title) in zip(axes, labels):
        xs = []
        for j, src in enumerate(["real", "virtual", "brightfield", "brightfield+virtual"]):
            r = rep.get(f"{src}|{lab}")
            if r is None:
                continue
            if lab == "viability":
                v, lo, hi = r["r2"], None, None
            else:
                a = r["classification"]["auc"]
                v, lo, hi = a["estimate"], a["ci_low"], a["ci_high"]
            ax.bar(j, v, 0.7, color=COL[src], edgecolor=SURF, linewidth=2)
            if lo is not None:
                ax.errorbar(j, v, yerr=[[v - lo], [hi - v]], color=INK2, capsize=3, lw=1)
            ax.text(j, (hi or v) + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
            xs.append(v)
        lo_ = 0.5 if lab == "treated_vs_dmso" else 0.85
        ax.set_ylim(lo_, 1.0 if lab == "treated_vs_dmso" else 1.012)
        ax.set_xticks([])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
    for src in COL:
        axes[0].bar(0, 0, color=COL[src], label=LBL[src])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    note(fig, "Best of 250 searched paths per source x label; out-of-fold AUC, 5-fold CV grouped by compound x 3 repeats, 95% CI = compound bootstrap.")
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(OUT / "fig_q2_summary.png", dpi=170)
    plt.close(fig)


# ================================================================================================ Q4
NICE = {"handcrafted": "hand-crafted", "dinov2_small": "DINOv2-S", "dinov2_base": "DINOv2-B",
        "concat_with_handcrafted": "+ hand-crafted", "correlation_filter_0.9": "corr. filter", "logistic_l1": "L1 logistic",
        "hist_gradient_boosting": "gradient boosting", "clahe+centre_crop": "CLAHE crop", "centre_crop": "crop",
        "single": "", "none": ""}


def nice_q4(k):
    if k.startswith("resnet18_ft"):
        return "Fine-tuned ResNet18 (all layers)"
    prep, rep, fus, red, model = k.split("/")
    parts = [NICE.get(rep, rep) + (" " + NICE[fus] if NICE.get(fus) else ""), NICE.get(prep, prep),
             NICE.get(red, red), NICE.get(model, model)]
    return ", ".join(p for p in parts if p)


def fig_q4():
    d = RES / "chips" / "q4"
    if not (d / "q4_assess.json").exists():
        return
    a = json.loads((d / "q4_assess.json").read_text())
    reps = a["reports"]
    order = sorted(reps, key=lambda k: reps[k]["auc"]["estimate"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), gridspec_kw={"width_ratios": [1.5, 1, 0.9]})
    ax = axes[0]
    for i, k in enumerate(order):
        e = reps[k]["auc"]
        base = k == a["baseline"]
        ax.barh(i, e["estimate"], color=GREY if base else COL["virtual"], edgecolor=SURF, linewidth=2)
        ax.errorbar(e["estimate"], i, xerr=[[e["estimate"] - e["ci_low"]], [e["ci_high"] - e["estimate"]]], color=INK2, capsize=3, lw=1)
        ax.text(0.505, i, f"{e['estimate']:.3f}", va="center", fontsize=8.5, color="white")
    ax.set_yticks(range(len(order)), [nice_q4(k) + (" (baseline)" if k == a["baseline"] else "") for k in order], fontsize=8.5)
    ax.set_xlim(0.5, 0.95)
    ax.set(title="Chip quality: best paths (AUC, 95% CI)", xlabel="ROC AUC, CV grouped by experiment")
    ax.grid(axis="y", visible=False)
    ax = axes[1]
    lo = a["leave_one_cell_type_out"]
    best, base = a["best"], a["baseline"]
    cells = list(lo[best]["per_group_auc"])
    for j, (k, c) in enumerate([(base, GREY), (best, COL["virtual"])]):
        ax.scatter([lo[k]["per_group_auc"][x] for x in cells], range(len(cells)), color=c, s=40, zorder=3)
    ax.set_yticks(range(len(cells)), cells)
    ax.axvline(0.5, color=INK2, lw=1, ls=":")
    ax.set(xlim=(0.4, 1.02), title="Unseen cell type (leave one out)", xlabel="AUC on the held-out cell type")
    ax = axes[2]
    cal = reps[best]["calibration"]
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
    ax.plot([c["mean_pred"] for c in cal], [c["frac_pos"] for c in cal], color=COL["virtual"], marker="o", ms=4)
    ax.set(xlim=(0, 1), ylim=(0, 1), title="Calibration (best path)", xlabel="predicted P(good)", ylabel="observed fraction good")
    note(fig, f"3,072 chip images, 59 experiments, 6 cell types. Search: {json.loads((d / 'q4_search_summary.json').read_text())['evaluated']} paths "
              "(TPE + successive halving). CIs: bootstrap over experiments.")
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=COL["virtual"], label="searched paths: best (DINOv2 + hand-crafted, L1 logistic) and runners-up"),
               Patch(color=GREY, label="baseline (hand-crafted + gradient boosting)"),
               Line2D([], [], color=INK2, ls=":", label="chance (AUC 0.5)"),
               Line2D([], [], color=INK2, ls="--", label="perfect calibration")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0.06, 1, 0.92))
    fig.savefig(OUT / "fig_q4_summary.png", dpi=170)
    plt.close(fig)


# ================================================================================================ examples
DATA_ROOT = Path(os.environ.get("VCP_DATA_ROOT", ROOT.parents[2] / "data" / "external"))
STAIN = ["DNA", "ER", "RNA", "AGP", "Mito"]


def _example(plate, data_dir, vdir, pick):
    """A held-out-compound field: real stains from the plate, virtual stains from the test-time cache (winner model)."""
    import os
    import sys
    os.environ["VCP_DATA"] = str(data_dir)
    sys.path.insert(0, str(ROOT))
    import importlib
    import config
    importlib.reload(config)
    import data as D
    importlib.reload(D)
    arr, idx = D.load_plate(plate)
    lo, hi = D.norm_params(plate)
    held = idx[(idx.split == "test")]
    i = int(held.index[pick])
    x = D.normalise(np.asarray(arr[i]), lo, hi)
    v = np.load(vdir / f"{plate}_{i:04d}.npy").astype(np.float32)
    return x[[0, 1, 2, 3, 4]], v, x[6], idx.loc[i, "pert_iname"]


def fig_examples():
    sets = [("A549", "BR00117017", DATA_ROOT / "cpjump1", RAW / "q3_v2" / "cache" / "virtual"),
            ("U2OS", "BR00117012", DATA_ROOT / "cpjump1_u2os48", RAW / "u2os" / "kaggle_q1" / "u2os_virtual")]
    sets = [s_ for s_ in sets if s_[3].exists() and (s_[2] / f"{s_[1]}.npy").exists()]
    if not sets:
        return
    fig, axes = plt.subplots(2 * len(sets), 6, figsize=(12, 4.2 * len(sets)))
    c0, w = 135, 270  # centre crop, so cells are visible
    for k, (name, plate, ddir, vdir) in enumerate(sets):
        real, virt, bf, cpd = _example(plate, ddir, vdir, 3)
        for r, (img, lab) in enumerate([(real, "real"), (virt, "virtual")]):
            ax = axes[2 * k + r, 0]
            if r == 0:
                b = bf[c0:c0 + w, c0:c0 + w]
                ax.imshow(b, cmap="gray", vmin=np.percentile(b, 1), vmax=np.percentile(b, 99.5))
                ax.set_title(f"{name}, {cpd}\nbright-field (input)", fontsize=9)
            ax.axis("off")
            for c in range(5):
                ref = real[c][c0:c0 + w, c0:c0 + w]
                lo_, hi_ = np.percentile(ref, [1, 99.7])  # same display range for real and virtual
                ax = axes[2 * k + r, c + 1]
                ax.imshow(img[c][c0:c0 + w, c0:c0 + w], cmap="gray", vmin=lo_, vmax=hi_)
                ax.set_title(f"{STAIN[c]} ({lab})", fontsize=9)
                ax.axis("off")
    note(fig, "Held-out compounds (never in training) on test plates never used for training or model choice. "
              "Real (top) vs virtual (bottom, winner model: seed ensemble + test-time augmentation). Centre crop of the field.\n"
              "Same display range for real and virtual. U2OS Mito is the stain that transfers worst (test r 0.65); here the virtual channel is much brighter than the real one.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "fig_examples.png", dpi=150)
    plt.close(fig)


def _composite(st, ref):
    rgb = []
    for c in (3, 1, 0):  # AGP -> red, ER -> green, DNA -> blue, scaled by the real stains
        lo_, hi_ = np.percentile(ref[c], [1, 99.7])
        rgb.append(np.clip((st[c] - lo_) / (hi_ - lo_ + 1e-6), 0, 1))
    return np.stack(rgb, -1)


def fig_q3_example():
    """One A549 test field segmented on real and on virtual stains with the winning Q3 path (Cellpose cyto3, all five channels)."""
    vdir = RAW / "q3_v2" / "cache" / "virtual"
    segf = RES / "a549" / "q3" / "seg_fields.json"
    if not (vdir.exists() and segf.exists()):
        return
    from skimage.segmentation import find_boundaries
    real, virt, bf, cpd = _example("BR00117017", DATA_ROOT / "cpjump1", vdir, 10)
    from sweep.q3_search import segment
    from sweep.q3_single_cell import match
    path = json.loads(segf.read_text())["seg_path"]
    mr, mv = segment(real, path), segment(virt, path)
    pairs, _ = match(mr, mv)
    f1 = 2 * len(pairs) / max(mr.max() + mv.max(), 1)
    c0, w = 100, 340
    crop = (slice(c0, c0 + w), slice(c0, c0 + w))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    b = bf[crop]
    axes[0].imshow(b, cmap="gray", vmin=np.percentile(b, 1), vmax=np.percentile(b, 99.5))
    axes[0].set_title("Bright-field (input)")
    for ax, st, m, lab in ((axes[1], real, mr, "real"), (axes[2], virt, mv, "virtual")):
        img = _composite(st, real)[crop].copy()
        img[find_boundaries(m[crop], mode="outer")] = (1, 1, 1)
        ax.imshow(img)
        ax.set_title(f"Cells on {lab} stains: {int(m.max())}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Q3: the same segmentation path on real and virtual stains (A549, {cpd}, unseen plate). "
                 f"Matched cells (IoU >= 0.5): {len(pairs)}, field F1 {f1:.2f}", fontsize=10.5)
    note(fig, "Cellpose cyto3 on all five channels (the winning path of the segmentation search). Colour: AGP red, ER green, DNA blue; "
              "white = cell outlines. Over all 1,536 test fields F1 = 0.979.")
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(OUT / "fig_q3_example.png", dpi=150)
    plt.close(fig)


# ================================================================================================ U2OS replication
def fig_replication():
    rjson = RES / "results.json"
    if not rjson.exists():
        return
    R = json.loads(rjson.read_text())
    if "u2_test_pixel_r" not in R:
        return
    num = lambda v: float(str(v).split()[0])  # noqa: E731
    A549, U2 = "#2a78d6", "#eb6834"
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), gridspec_kw={"width_ratios": [1, 1.25, 1.1]})
    ax = axes[0]
    x = np.arange(5)
    for j, (pre, col, lab) in enumerate([("q1_test", A549, "A549 (main study)"), ("u2_test", U2, "U2OS (replication)")]):
        vals = [num(R[f"{pre}_pixel_r"][c]) for c in STAIN]
        ax.bar(x + (j - 0.5) * 0.38, vals, 0.38, color=col, edgecolor=SURF)
    ax.set_xticks(x, STAIN)
    ax.set(ylim=(0.5, 1.0), title="Q1 virtual staining (test)", ylabel="pixel Pearson r")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    labs = [("toxic_05", "kills >50%"), ("toxic_08", "kills >20%"), ("viability", "viability R²"), ("treated_vs_dmso", "non-toxic vs DMSO")]
    x = np.arange(len(labs))
    for j, (pre, col) in enumerate([("q2", A549), ("u2_q2", U2)]):
        for src, hatch, off in (("virtual", None, -0.2), ("brightfield", "///", 0.2)):
            vals = [num(R[f"{pre}_{l}_{src}"]) for l, _ in labs]
            ax.bar(x + off + (j - 0.5) * 0.19, vals, 0.19, color=col if src == "virtual" else SURF, edgecolor=col, hatch=hatch, lw=1.2)
    ax.set_xticks(x, [b for _, b in labs], fontsize=8.5)
    ax.set(ylim=(0.35, 1.02), title="Q2 drug response", ylabel="AUC (R² for viability)")
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.grid(axis="x", visible=False)
    ax = axes[2]
    mets = [("f1", "cell F1"), ("ap", "mask AP"), ("feature_ccc", "feature CCC"), ("count_ccc", "count CCC"), ("ari", "states ARI")]
    x = np.arange(len(mets))
    for j, (pre, col, lab) in enumerate([("q3", A549, "A549"), ("u2_q3", U2, "U2OS")]):
        ax.bar(x + (j - 0.5) * 0.38, [num(R[f"{pre}_{m}"]) for m, _ in mets], 0.38, color=col, edgecolor=SURF)
    ax.set_xticks(x, [b.replace(" ", "\n") for _, b in mets], fontsize=8.5)
    ax.set(ylim=(0.4, 1.0), title="Q3 single cells: virtual vs real")
    ax.grid(axis="x", visible=False)
    note(fig, "U2OS = second dataset, separate study: the A549 winning paths retrained/applied on U2OS 48 h (train BR00117011, "
              "test BR00117012), nothing re-chosen. One U2OS test plate.")
    from matplotlib.patches import Patch
    handles = [Patch(color=A549, label="A549 (main study)"), Patch(color=U2, label="U2OS (replication)"),
               Patch(facecolor=GREY, edgecolor=GREY, label="Q2: solid = virtual stains"),
               Patch(facecolor=SURF, edgecolor=INK2, hatch="///", label="Q2: hatched = bright-field only")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(OUT / "fig_replication.png", dpi=170)
    plt.close(fig)


def fig_system():
    """Figure 1: the deployed system (figures/fig_system_deploy.dot) with a zoom on one question box showing what its ⚙ stands
    for, the model-selection procedure used inside every question (figures/fig_system_search.dot). Needs Graphviz ('dot')."""
    import shutil
    import subprocess
    from matplotlib.patches import ConnectionPatch, Rectangle
    if not shutil.which("dot"):
        print("skip fig_system: Graphviz 'dot' not installed")
        return
    dpi, pad = 200, 0.12
    imgs = {}
    for part in ("deploy", "search"):
        src = OUT / f"fig_system_{part}.dot"
        png = OUT / f".fig_system_{part}.png"
        subprocess.run(["dot", "-Tpng", f"-Gdpi={dpi}", str(src), "-o", str(png)], check=True)
        imgs[part] = plt.imread(png)
        png.unlink()
    plain = subprocess.run(["dot", "-Tplain", str(OUT / "fig_system_deploy.dot")], capture_output=True, text=True, check=True).stdout
    gh = float(plain.splitlines()[0].split()[3])
    x, y, w, h = (float(v) for v in next(l for l in plain.splitlines() if l.startswith("node q4 ")).split()[2:6])
    # zoomed box in pixels of the top image (Graphviz: inches, origin bottom-left; image: origin top-left, plus the pad)
    bx0, bx1 = (x - w / 2 + pad) * dpi, (x + w / 2 + pad) * dpi
    by0, by1 = (gh - y - h / 2 + pad) * dpi, (gh - y + h / 2 + pad) * dpi
    top, bot = imgs["deploy"], imgs["search"]
    band = np.ones((90, bot.shape[1], bot.shape[2]), dtype=bot.dtype)  # room for panel B's title inside its box
    band[..., :3] = bot[2, 2, :3]
    bot = np.vstack([band, bot])
    W = max(top.shape[1], bot.shape[1]) / dpi
    fig = plt.figure(figsize=(W, (top.shape[0] + bot.shape[0]) / dpi + 0.9))
    hs = [top.shape[0], 0.9 * dpi * 0.55, bot.shape[0]]
    gs = fig.add_gridspec(3, 1, height_ratios=hs, hspace=0)
    a1, a3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[2])
    a1.imshow(top)
    a3.imshow(bot)
    for a in (a1, a3):
        a.set_axis_off()
    a1.set_title("A. The system on new images; ⚙ = a model chosen by its own path search", loc="left", fontsize=13, fontweight="normal",
                 color=INK)
    a3.text(30, 50, "B. Inside each ⚙ (zoom on Q4; the same procedure for Q1, Q2, Q3 and Q4)", fontsize=13, color=INK, va="center")
    a1.add_patch(Rectangle((bx0 - 10, by0 - 10), bx1 - bx0 + 20, by1 - by0 + 20, fill=False, ec=INK2, lw=1.4, ls="--"))
    hb, wb = bot.shape[0], bot.shape[1]
    for (xa, xb) in ((bx0 - 10, 2), (bx1 + 10, wb - 2)):
        fig.add_artist(ConnectionPatch((xa, by1 + 10), (xb, 2), "data", "data", axesA=a1, axesB=a3, color=GREY, lw=1.1, ls="--"))
    a3.add_patch(Rectangle((1, 1), wb - 3, hb - 3, fill=False, ec=INK2, lw=1.4, ls="--"))
    fig.savefig(OUT / "fig_system.png", dpi=dpi, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


def fig_fit():
    """Overfitting and underfitting checks, from logged scores only: (A) Q1 train - validation gap at each checkpoint,
    (B) the best path's validation score at 25/50/100% of the data per question, (C) selection score vs held-out score."""
    db = RES / "a549" / "q1_search" / "study.db"
    paths = {"Q2 non-toxic AUC": RES / "a549" / "q2" / "searches" / "q2_virtual_treated_vs_dmso_paths.csv",
             "Q3 segmentation F1": RES / "a549" / "q3" / "seg" / "q3_seg_paths.csv",
             "Q4 chip AUC": RES / "chips" / "q4" / "q4_search_paths.csv"}
    final = ROOT / "models" / "final_models.json"
    if not (db.exists() and final.exists() and all(p.exists() for p in paths.values())):
        print("skip fig_fit: missing inputs")
        return
    import optuna
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    study = optuna.load_study(study_name="q1", storage=f"sqlite:///{db}")
    trials = [t for t in study.trials if t.state.name == "COMPLETE" and "u2os" not in t.params.get("data", "")]
    R, F = json.loads((RES / "results.json").read_text()), json.loads(final.read_text())
    num = lambda v: float(str(v).split()[0])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    frac = [25, 50, 100]

    ax = axes[0]  # A: Q1 train - validation gap
    best = max(trials, key=lambda t: t.value)
    off = []
    for t in trials:
        g = [t.user_attrs.get(f"p{i}_gap") for i in (1, 2, 3)]
        if None in g:
            continue
        if max(abs(x) for x in g) > 0.01:  # the pretrained-encoder path: far off this scale, reported in text
            off.append(g)
            continue
        ax.plot(frac, g, color=COL["virtual"] if t is best else GREY, lw=2 if t is best else 1, alpha=1 if t is best else 0.6,
                marker="o", ms=3, zorder=3 if t is best else 2)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set(xticks=frac, xlabel="training data used (%)", ylabel="train r − validation r", title="A. Q1 overfitting gap",
           ylim=(-0.001, 0.008))
    if off:
        g = off[0]
        ax.text(0.03, 0.97, f"pretrained-encoder path, off scale:\n{g[0]:+.3f}, {g[1]:+.3f}, {g[2]:+.3f}", transform=ax.transAxes,
                va="top", fontsize=7.5, color=INK2)
    ax.plot([], [], color=COL["virtual"], lw=2, label="best path")
    ax.plot([], [], color=GREY, label="other U-Net paths")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, bbox_to_anchor=(-0.02, -0.18))

    ax = axes[1]  # B: best path's validation score as the data grow (gain over its 25% score)
    curves = {"Q1 pixel r": [best.intermediate_values[i] for i in (1, 2, 3)]}
    for k, f in paths.items():
        d = pd.read_csv(f)
        d = d[d.state == "COMPLETE"].sort_values("value")
        r = d.iloc[-1]
        curves[k] = [r.score_25, r.score_50, r.score_100]
    colors = ["#eb6834", "#2a78d6", "#4a3aa7", "#1baf7a"]
    for (k, c), col in zip(curves.items(), colors):
        y = [v - c[0] for v in c]
        ax.plot(frac, y, color=col, lw=1.8, marker="o", ms=3.5, label=f"{k} ({c[0]:.3f} → {c[2]:.3f})")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set(xticks=frac, xlabel="data used (%)", ylabel="validation score − score at 25%", title="B. Best path: gain with more data")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, bbox_to_anchor=(-0.02, -0.18), ncol=1)

    ax = axes[2]  # C: score used to choose the path vs score on data never used to fit it
    rows = [("Q1 pixel r (val → test)", float(R["q1_inference_val"]["seed_ensemble+tta"]),
             float(np.mean([float(v) for v in R["q1_test_pixel_r"].values()]))),
            ("Q2 toxicity > 50% AUC", num(R["q2_toxic_05_virtual"]), F["q2"]["toxic_0.5"]["held_out_AUC"]),
            ("Q2 toxicity > 20% AUC", num(R["q2_toxic_08_virtual"]), F["q2"]["toxic_0.8"]["held_out_AUC"]),
            ("Q2 viability R²", num(R["q2_viability_virtual"]), F["q2"]["viability"]["held_out_R2"]),
            ("Q2 non-toxic AUC", num(R["q2_treated_vs_dmso_virtual"]), F["q2"]["treated_vs_dmso"]["held_out_AUC"]),
            ("Q3 cell states ARI", float(R["q3_ari"]), F["q3"]["held_out_ARI_real_vs_virtual_states"]),
            ("Q4 chip AUC", num(R["q4_best"]), F["q4"]["held_out_AUC"])]
    for i, (k, a, b) in enumerate(rows[::-1]):
        ax.plot([a, b], [i, i], color=GREY, lw=1.5, zorder=1)
        ax.scatter([a], [i], s=36, facecolor=SURF, edgecolor=INK2, zorder=2)
        ax.scatter([b], [i], s=36, color=COL["virtual"], zorder=3)
        ax.text(1.005, i, f"{a:.3f} → {b:.3f}", va="center", fontsize=7.5, color=INK2, transform=ax.get_yaxis_transform())
    ax.set_yticks(range(len(rows)), [r[0] for r in rows[::-1]], fontsize=8.5)
    ax.set(xlabel="score", title="C. Selection vs final-fit check", xlim=(0.65, 1.01))
    ax.grid(axis="y", visible=False)
    ax.scatter([], [], s=36, facecolor=SURF, edgecolor=INK2, label="used to choose (validation / cross-validation)")
    ax.scatter([], [], s=36, color=COL["virtual"], label="excluded from the final fit (used earlier in selection)")
    ax.legend(loc="upper left", fontsize=7.5, frameon=False, bbox_to_anchor=(-0.02, -0.18))
    fig.subplots_adjust(left=0.06, right=0.93, top=0.92, bottom=0.36, wspace=0.62)
    fig.savefig(OUT / "fig_fit.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    for f in (fig_system, fig_q1_search, fig_fit, fig_q2, fig_q4, fig_examples, fig_q3_example, fig_replication):
        f()
        print("done", f.__name__)
