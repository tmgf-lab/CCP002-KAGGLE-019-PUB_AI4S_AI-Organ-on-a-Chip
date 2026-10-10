# Web demo of the end-to-end system (Gradio)

Predicts the five Cell Painting stains (DNA, ER, RNA, AGP, Mito) from bright-field images, so cells can be profiled without
staining. The model is the winner of the guided search over U-Net variants (width 64, bilinear up-sampling, 2 seeds), tested once
on unseen plates: pixel r 0.93–0.98 on held-out compounds, nuclei per well r 0.991.

```bash
pip install -r requirements.txt        # CPU is enough (adds gradio to the repository requirements)
python app.py                          # uses the repository code; downloads the winner weights on first run (weights.py)
```

- **Tab 1** streams a real field of plate BR00117019 (never used to fit any model) from the public JUMP Cell Painting pilot
  and runs the system on it: **Q1** virtual stains (real stains shown for comparison), **Q2** drug response (per-field
  estimate: toxicity calls, viability, drug effect) and **Q3** cells (outlines, count, cell-state shares). About 25 s on a CPU.
- **Tab 2** runs a single-plane model on your own bright-field image: experimental, not validated outside CPJUMP1.
- **Tab 3** runs **Q4** chip quality on an organ-on-a-chip photo; the examples come from the 6 chip experiments held out of
  the Q4 fit. About 2 s.

For whole wells (2 fields averaged, the plate's own DMSO controls) and a report, use `python vcp_system.py` at the repository root.

The demo is shown working in the project video. Data and licences: model and code MIT; training data Cell Painting Gallery
`cpg0000-jump-pilot` (CC0); example chip images *Organ-on-a-Chip (OOC) Image Dataset*, Zenodo 10203721, CC BY 4.0
(doi:10.3390/data9020028), centre crops.
