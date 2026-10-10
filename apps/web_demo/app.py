"""Virtual Cell Painting: Gradio web demo of the end-to-end system (run locally: python app.py, open the printed URL).

Tab 1 streams a field of the public JUMP Cell Painting pilot (plate BR00117019, never used to fit any model) and runs
      Q1 (virtual stains), Q2 (drug response, per-field estimate) and Q3 (cells and cell states): vcp_system.System.
Tab 2 applies the single-plane model to an uploaded bright-field image (experimental, not validated).
Tab 3 runs Q4 (chip quality) on an organ-on-a-chip photo; examples come from experiments held out of the Q4 fit.
"""
import io
import json
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]  # the repository: shared code, the Q1 winner, the legacy single-plane model
sys.path.insert(0, str(ROOT))
import config as C  # noqa: E402
from data import load_model, predict, predict_multi  # noqa: E402
from download import build_wells, fetch, load_site  # noqa: E402
from evaluate import nuclei  # noqa: E402

torch.set_num_threads(2)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
# the Q1 winner: U-Net width 64, bilinear, 2 seeds (chosen on validation fields; tested once on the unseen plates)
from vcp_system import System, composite as cells_composite  # noqa: E402
SYS = System(device=DEV, tta=False)  # Q1 winner (2 seeds) + the final Q2, Q3, Q4 models (models/)
MODELS, CFG = SYS.models, SYS.cfg
STATE_NAMES = SYS.state_names()
CHIP_EXAMPLES = sorted(str(p) for p in (Path(__file__).parent / "examples" / "chips").glob("*.png"))
MODEL1, CFG1 = load_model(ROOT / "legacy" / "runs" / "chip1", dev=DEV)       # 1 plane, robustness-augmented (for uploads)
NORM = json.loads((ROOT / "runs" / "winner" / "norm.json").read_text())
TGT = [C.CHANNELS[c] for c in C.TARGET_CH]
WELLS = build_wells()
SPLIT_NAME = {"test": "held out of training", "val": "validation", "train": "seen in training", "control": "control"}
CHOICES = ["DMSO"] + sorted(
    (f"{n}  [{SPLIT_NAME[s]}]" for n, s in WELLS[WELLS.pert_iname != "DMSO"][["pert_iname", "split"]].drop_duplicates().itertuples(index=False)),
    key=lambda t: (not t.endswith("[held out of training]"), t.lower()))
PLATES = {"BR00117019 (never used to fit any model)": "BR00117019"}
MAX_SIDE = 1024


@lru_cache(maxsize=4)
def load_data(plate):
    return pd.read_csv(io.BytesIO(fetch(f"{C.S3}/{C.PILOT}/load_data_csv/{C.BATCH}/{plate}/load_data.csv")))


def composite(stains, ref=None):
    """RGB: AGP -> red, ER -> green, DNA -> blue. Scaled by `ref` (the real stains) when given."""
    ref = stains if ref is None else ref
    rgb = []
    for c in (3, 1, 0):
        lo, hi = np.percentile(ref[c], [1, 99.7])
        rgb.append(np.clip((stains[c] - lo) / (hi - lo + 1e-6), 0, 1))
    return np.stack(rgb, -1)


def panel(rows, title):
    """rows: list of (label, [(name, image, (vmin, vmax) or None)])"""
    n = max(len(r[1]) for r in rows)
    fig, ax = plt.subplots(len(rows), n, figsize=(2.6 * n, 3.0 * len(rows)), squeeze=False)
    for i, (label, imgs) in enumerate(rows):
        for j in range(n):
            a = ax[i, j]
            a.set_xticks([]); a.set_yticks([])
            if j >= len(imgs) or imgs[j] is None:
                a.axis("off")
                continue
            name, img, rng = imgs[j]
            if img.ndim == 3:
                a.imshow(img)
            else:
                vmin, vmax = rng or np.percentile(img, [1, 99.7])
                a.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            a.set_title(name, fontsize=10)
        first = next(j for j, im in enumerate(imgs) if im is not None)
        ax[i, first].set_ylabel(label, fontsize=11)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(h_pad=2.2)
    return fig


def run_jump(choice, plate_label, site, tta=False):
    name = choice.split("  [")[0]
    plate = PLATES[plate_label]
    info = WELLS[WELLS.pert_iname == name].iloc[0]
    ld = load_data(plate)
    row = ld[(ld.Metadata_Well == info.well) & (ld.Metadata_Site == int(site))].iloc[0]
    lo, hi = np.array(NORM[plate]["lo"], np.float32)[:, None, None], np.array(NORM[plate]["hi"], np.float32)[:, None, None]
    x = (load_site(row).astype(np.float32) - lo) / (hi - lo)
    SYS.tta = bool(tta)
    virt = SYS.stain(x)                                    # Q1: 2-seed ensemble (+ 8 flips/rotations with TTA)
    real = x[C.TARGET_CH]
    mask, states = SYS.cells(virt)                         # Q3 on the virtual stains
    q2 = SYS.drug_response(pd.DataFrame([SYS.field_features(virt)])).iloc[0]   # Q2, one field
    n_real, n_virt = nuclei(real[0])[0], nuclei(virt[0])[0]
    r = [np.corrcoef(real[c].ravel(), virt[c].ravel())[0, 1] for c in range(5)]
    rngs = [tuple(np.percentile(real[c], [1, 99.7])) for c in range(5)]
    bf = x[C.INPUT_CH[1]]
    fig = panel([
        ("Real", [("Bright-field (input)", bf, None), *[(f"{TGT[c]}", real[c], rngs[c]) for c in range(5)], ("Composite", composite(real), None)]),
        ("Virtual", [None, *[(f"{TGT[c]}  r={r[c]:.2f}", virt[c], rngs[c]) for c in range(5)],
                     ("Q3 cells (outlines)", cells_composite(virt, real, mask), None)]),
    ], f"{name} | {plate} well {info.well} field {int(site)}")
    target = info.target if isinstance(info.target, str) else "-"
    md = (f"**{name}** (target: {target}; {SPLIT_NAME[info.split]})  \n"
          f"**Nuclei in field:** real stain **{n_real}**, virtual stain **{n_virt}**  \n"
          f"**Pixel agreement (Pearson r):** " + ", ".join(f"{TGT[c]} {r[c]:.2f}" for c in range(5)) +
          f"  \n*Q1 model: 2-seed ensemble{' + test-time augmentation' if tta else ''}.*  \n\n"
          f"**Q2 drug response** (an estimate: one field, and the normalisation uses the calibration plate's DMSO wells; "
          f"`vcp_system.py` averages 2 fields per well and uses the new plate's own DMSO wells): kills >50% "
          f"**{'yes' if q2['toxic_0.5_call'] else 'no'}** (score {q2['toxic_0.5_score']:.2f}), kills >20% "
          f"**{'yes' if q2['toxic_0.8_call'] else 'no'}** (score {q2['toxic_0.8_score']:.2f}), viability "
          f"**{q2['viability_estimate']:.2f}** of DMSO, non-toxic drug-effect score {q2['treated_vs_dmso_score']:.2f} "
          f"({'low confidence, held-out AUC 0.73' if not q2['toxic_0.8_call'] else 'not applicable: the well looks toxic'})  \n"
          f"**Q3 single cells** (on the virtual stains): **{int(mask.max())}** cells; states: "
          + (", ".join(f"{n} {np.mean(states == k):.0%}" for k, n in enumerate(STATE_NAMES)) if len(states) else "no cells")
          + f"  \n*States (names ours): {SYS.state_legend()}.*")
    if n_real > 150 and n_virt < 20:
        md += ("  \n*Note: on near-empty fields (all cells killed) the simple nuclei counter can count background "
               "shading on the real stain. Look at the images.*")
    return fig, md


def run_upload(img, scale):
    if img is None:
        raise gr.Error("Upload a bright-field image first.")
    im = Image.fromarray(img).convert("F")
    w, h = round(im.width * scale), round(im.height * scale)
    f = min(1.0, MAX_SIDE / max(w, h))
    im = im.resize((max(16, round(w * f)), max(16, round(h * f))), Image.BILINEAR)
    bf = np.asarray(im, np.float32)
    virt = predict(MODEL1, CFG1, bf[None], DEV)
    n_virt = nuclei(virt[0])[0]
    fig = panel([("Virtual", [("Bright-field (input)", bf, None), *[(TGT[c], virt[c], None) for c in range(5)],
                              ("Composite", composite(virt), None)])],
                f"{bf.shape[1]}x{bf.shape[0]} px after rescaling")
    md = (f"**Virtual nuclei counted:** {n_virt}  \n"
          "*Experimental: this single-plane model was trained only on A549 cells in 384-well plates (CPJUMP1). "
          "On organ-on-a-chip images it was evaluated zero-shot and is **not reliable**: about a quarter of images "
          "looked plausible, and dense monolayers produce texture instead of cells. Treat the output as an illustration, "
          "not a measurement.*")
    return fig, md


def run_chip(img):
    if img is None:
        raise gr.Error("Choose or upload a chip photo first.")
    p = SYS.chip_quality(Image.fromarray(img))
    return (f"### P(good chip) = {p:.3f} → **{'good' if p >= 0.5 else 'bad'}**\n"
            "Q4 model: DINOv2-small + hand-crafted image features, correlation filter, L1 logistic (the search winner), fitted "
            "on 53 experiments. The examples are 6 random images from the 6 held-out experiments (one per cell type), never "
            "used for fitting; the model is not always right on them (held-out AUC 0.80).")


INTRO = """# Virtual Cell Painting: the end-to-end system
From **bright-field images alone**: **Q1** predicts the five Cell Painting stains (DNA, ER, RNA, AGP = actin/Golgi/membrane,
Mito), **Q2** reads the drug response (toxicity, viability, non-toxic drug effect), **Q3** finds the cells and their states, and
**Q4** checks the quality of an organ-on-a-chip photo. No staining, so the cells stay alive. Team tmgf,
*AI4S Open Innovation: AI for Life Science* (End-to-End System).

**On held-out compounds and unseen plates (A549):**
- virtual vs real stains agree pixel-wise at **r = 0.93–0.98**;
- nuclei per well agree at **r = 0.991**;
- virtual stains close **65%** of the drug-response gap between bright-field and real stains (**85%** for compounds never seen in training).

Model: the winner of a guided search over U-Net variants (width 64, bilinear up-sampling, 2 seeds). Replicated on a second cell
line (U2OS): four stains r 0.89–0.95, mitochondria 0.65.

[Code & report](https://github.com/tmgf-lab/CCP002-KAGGLE-019-PUB_AI4S_AI-Organ-on-a-Chip) · Video: see the Kaggle Writeup<!--YOUTUBE--> · [Kaggle notebook](https://www.kaggle.com/code/tiagomgfernandes/organ-on-a-chip-virtual-staining-walkthrough) · Data: JUMP Cell Painting pilot (CPJUMP1, CC0)"""

with gr.Blocks(title="Virtual Cell Painting") as demo:
    gr.Markdown(INTRO)
    with gr.Tab("1 · Cells: stains, drug response, single cells (Q1–Q3)"):
        gr.Markdown("Streams one field (8 TIFFs, about 5 s) from the public Cell Painting Gallery. The model sees **only the 3 bright-field planes**. "
                    "This plate was never used to fit any model; compounds marked *held out of training* were never seen at all.")
        with gr.Row():
            comp = gr.Dropdown(CHOICES, value="arcyriaflavin-a  [held out of training]", label="Compound (A549 cells, 48 h)")
            plate = gr.Dropdown(list(PLATES), value=list(PLATES)[0], label="Plate")
            site = gr.Slider(1, 16, value=1, step=1, label="Field (1-16)")
        tta = gr.Checkbox(False, label="Test-time augmentation (the validated best mode, about 45 s on a CPU; the default takes about 6 s)")
        go = gr.Button("Stain virtually", variant="primary")
        out_md = gr.Markdown()
        out_plot = gr.Plot()
        go.click(run_jump, [comp, plate, site, tta], [out_plot, out_md])
        gr.Examples([["DMSO", list(PLATES)[0], 1], ["PP-121  [held out of training]", list(PLATES)[0], 1],
                     ["arcyriaflavin-a  [held out of training]", list(PLATES)[0], 1]], [comp, plate, site])
    with gr.Tab("2 · Your bright-field image (experimental)"):
        gr.Markdown("Upload a grey-scale bright-field image. The model expects cells of about 15-25 px diameter (about 1.3 µm/px). "
                    "Use **scale** to match: e.g. 2 for images at about 2.6 µm/px. Large images are capped at 1024 px.  \n"
                    "**Not validated outside CPJUMP1**: see the note below each result.")
        with gr.Row():
            up = gr.Image(label="Bright-field image", image_mode="L", type="numpy")
            with gr.Column():
                scale = gr.Slider(0.5, 3.0, value=1.0, step=0.1, label="Scale")
                go2 = gr.Button("Stain virtually", variant="primary")
        out_md2 = gr.Markdown()
        out_plot2 = gr.Plot()
        go2.click(run_upload, [up, scale], [out_plot2, out_md2])
        gr.Examples([["examples/jump_DMSO_brightfield.png", 1.0],
                     ["examples/chip_230525_112.png", 2.0],
                     ["examples/chip_230425_49.png", 2.0]], [up, scale],
                    label="Examples: a CPJUMP1 field (CC0); two organ-on-a-chip images from Zenodo 10203721 (CC BY 4.0), "
                          "one that works and one that fails")

    with gr.Tab("3 · Chip quality (Q4)"):
        gr.Markdown("Organ-on-a-chip photo (bright-field, whole chip). The model looks at the centre region (1024 × 768 px).")
        with gr.Row():
            chip = gr.Image(label="Chip photo", image_mode="L", type="numpy")
            with gr.Column():
                go3 = gr.Button("Check chip quality", variant="primary")
                out_md3 = gr.Markdown()
        go3.click(run_chip, [chip], [out_md3])
        gr.Examples([[p] for p in CHIP_EXAMPLES], [chip],
                    label="6 random held-out chips, file name = true label (Organ-on-a-Chip Image Dataset, Zenodo 10203721, CC BY 4.0; centre crops)")

if __name__ == "__main__":
    demo.launch()
