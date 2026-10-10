# Virtual Cell Painting for organ-on-a-chip

Label-free, non-destructive readout of organ-on-a-chip experiments: predict Cell Painting stains from bright-field, then use them for drug
toxicity/response, single-cell analysis, and an automatic chip quality check. Submission of team **tmgf** (tiagomgfernandes) to
*AI4S Open Innovation: AI for Life Science*, category **End-to-End System**.

**Video:** [https://youtu.be/ddly-1rWOGY](https://youtu.be/ddly-1rWOGY) · **Notebook:** https://www.kaggle.com/code/tiagomgfernandes/organ-on-a-chip-virtual-staining-walkthrough · **Report:** [`docs/technical_report.pdf`](docs/technical_report.pdf)

## Quick start (CPU is enough)

```bash
python3.11 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt          # Python 3.11; CPU-only torch is fine (exact versions of the reported runs)
python -m ipykernel install --user --name vcp --display-name "Python (vcp, organ-on-a-chip)"   # kernel for the notebook
python weights.py                        # downloads the final model (2 x 121 MB), verifies checksums
python vcp_system.py --plate BR00117019 --wells B16 B04 A01 --dmso 4 --chips apps/web_demo/examples/chips/*.png --out reports/demo
                                         # END-TO-END: Q1 virtual stains -> Q2 drug response + Q3 cells per well; Q4 chip quality
python demo.py --compound PP-121         # Q1 only, one field -> demo.png
cd apps/web_demo && python app.py        # web demo of the system (Q1-Q4), opens in the browser
jupyter lab notebooks/walkthrough.ipynb  # walkthrough of Q1-Q4 with small real runs + all saved results (under 20 min on CPU);
                                         # needs Jupyter (pip install jupyterlab) or VS Code; select the kernel
                                         # "Python (vcp, organ-on-a-chip)"
```

## Inputs and outputs

| | Input | Output |
|---|---|---|
| **`vcp_system.py`** | CPJUMP1 wells (3 bright-field z-planes per field, streamed from the public archive) + organ-on-a-chip images | per well: Q2 toxicity calls, viability estimate, drug-effect score; Q3 cells per field and cell-state shares; per chip: Q4 P(good); `report.html` + `report.json` |
| `demo.py` | a CPJUMP1 field (3 bright-field z-planes), streamed from the public S3 archive | 5 virtual stains (DNA, ER, RNA, AGP, Mito), nuclei counts real vs virtual, pixel r per stain, PNG |
| Q1 `sweep/q1_translation.py` | packed plates from `download.py` (uint16, 8 channels, 540 × 540 per field) | search log (`study.db`), trained models (`best.pt`), `q1_winner.json` |
| `evaluate.py --winner` | the winner + test plates | `report.json` (pixel, nuclei agreement, phenotypic activity), `site_features.csv` |
| Q2 `sweep/q2_search.py` | `site_features.csv` (per-field features from real, virtual and bright-field images) | `*_paths.csv` (every path, every checkpoint), `q2_report.json` |
| Q3 `sweep/q3_search.py` | real and virtual stains | segmentation / state / classifier search logs, `cells.parquet`, `q3_report.json` |
| Q4 `sweep/q4_chip_qc.py` | OoC images (Zenodo 10203721) | embeddings, search log, `q4_assess.json` |

## Results

| Question | Best path (guided search, A549) | A549 | U2OS replication (same paths) |
|---|---|---|---|
| Q1 virtual staining | U-Net w64, bilinear, 2 seeds + TTA (30 paths) | pixel r DNA 0.930, ER 0.969, RNA 0.977, AGP 0.946, Mito 0.949 | DNA 0.920, ER 0.917, RNA 0.953, AGP 0.896, Mito 0.654 |
| Q2 drug response | best of 3,912 paths | non-toxic drugs: virtual 0.699 [0.645, 0.749], bright-field 0.567 [0.512, 0.623], real 0.714 [0.659, 0.764] | toxicity replicates; non-toxic effect not detectable (real 0.516 [0.431, 0.603]) |
| Q3 single cells | Cellpose cyto3, all channels (31 + 65 paths) | F1 0.979 [0.978, 0.980], states ARI 0.745 | F1 0.951 [0.939, 0.962], ARI 0.605 |
| Q4 chip QC | DINOv2-S + hand-crafted, L1 logistic (1,003 paths) | AUC 0.865 [0.833, 0.893]; unseen cell types 0.853 | — |
| **System (final models)** | winning paths fitted on calibration data | plate excluded from the final fit (used earlier to choose the paths): Q2 AUC 0.998 / 0.995 (toxic 50 / 20%), R² 0.942 (viability), AUC 0.727 (drug effect); Q3 ARI 0.714 | Q4: 6 held-out chip experiments, AUC 0.800 |

Every number above and in the report is in `results/` (`results/results.json` collects them; `results/a549`, `results/u2os`,
`results/chips` hold the reports and every search path).

## How the analysis is organised

Each question is a tree of options (processing → model → exploration → evaluation) in `sweep/configs/*.yaml`; each dataset is a
separate study. `sweep/engine.py` searches the tree: balanced warm-up, Optuna TPE, nested 25% → 50% → 100% data checkpoints with
successive halving, an option-exclusion rule (4 bad results), duplicate skipping, fixed budgets.

| File | Purpose |
|---|---|
| **`vcp_system.py`** | **the end-to-end system** (one command: images in, report with Q1–Q4 out) |
| `models/` | the final Q2 / Q3 / Q4 models (winning paths fitted on calibration data; checks on data excluded from the final fit in `final_models.json`) |
| `tools/fit_final_models.py` | refits those models from the winning paths |
| `demo.py`, `weights.py`, `show.py` | Q1-only demo; weights download; short summaries of any result folder |
| `notebooks/walkthrough.ipynb` | walkthrough + live demo (built by `notebooks/build_walkthrough.py`) |
| `sweep/engine.py`, `sweep/stats.py` | search engine; statistics (bootstrap CIs, DeLong, Holm/BH, CCC, ICC, Bland–Altman, Bayesian bootstrap) |
| `sweep/q1_translation.py` | Q1 search, final seeds, variants, inference choice, `replicate` (winner on another dataset) |
| `sweep/q2_search.py`, `sweep/q2_models.py` | Q2: 16 searches (source × label), assessment, `--paths-from` replication |
| `sweep/q3_search.py` | Q3: virtual stains → segmentation search → cell features → state search → assessment |
| `sweep/q4_chip_qc.py`, `sweep/q4_parts.py` | Q4: embeddings (10 representations), classifier search, fine-tuning, assessment |
| `model.py`, `data.py`, `evaluate.py`, `download.py` | U-Net variants, data loading, test evaluation, data download |
| `legacy/` | the starting solution before the guided search (baseline model, early experiments, their scripts and figures) |
| `kaggle_jobs/build.py` | writes a Kaggle notebook for each stage (to run the searches on free Kaggle GPUs) |
| `apps/web_demo`, `apps/kaggle` | web demo (Gradio; `cd apps/web_demo && python app.py`, opens in the browser) and the Kaggle notebook package (`push.sh`) |
| `demo_run/` | the scripts that recorded the real runs in the video and built it (`record_all.sh`, `record_lab.py`, `build_video.py`) |
| `tools/` | `collect_results.py` (results/ → results.json), `report_figures.py` (figures/), `audit_numbers.py` (checks every number of the report against the run outputs → `docs/number_sources.md`), `q4_exhaustive.py` (runs the whole searched Q4 tree as a check of the guided search), `fill_links.py` (public links) |
| `docs/` | technical report (`technical_report.md` → `bash docs/build_pdf.sh`, or `docs\build_pdf.bat` on Windows → PDF) |
| `results/`, `figures/` | every score, report and search log; report figures |
| `video/` (not in git) | the demo video and its English / Chinese captions |

## Reproduce the full results

```bash
python download.py                                                   # CPJUMP1 A549 plates (~20 GB)
python -m sweep.q1_translation init && python -m sweep.q1_translation search --gpu 0     # + final, variants, choose
python evaluate.py --winner runs/q1_translation/q1_winner.json --out runs/q1_test
python -m sweep.q2_search all --features runs/q1_test/site_features.csv
python -m sweep.q3_search predict --winner runs/q1_translation/q1_winner.json && python -m sweep.q3_search segsearch
python -m sweep.q3_search features && python -m sweep.q3_search downstream && python -m sweep.q3_search supervised && python -m sweep.q3_search assess
python -m sweep.q4_chip_qc embed && python -m sweep.q4_chip_qc search && python -m sweep.q4_chip_qc assess
VCP_TRAIN_PLATES=BR00117011 VCP_TEST_PLATES=BR00117012 python -m sweep.q1_translation replicate --winner ...   # U2OS
```

**Hardware and services.** Only free, public services: the Cell Painting Gallery (S3, no login) and Zenodo for data, the GitHub
release for the weights. Everything runs locally with the commands above (no GPU needed for the demo, the system, the notebook
and the assessments; training needs one). The long searches were run as Kaggle notebooks (free T4 GPUs): `kaggle_jobs/build.py`
writes them, but they read intermediate inputs kept private in the team's Kaggle account, so treat that file as a record of how
the runs were launched; the commands above are the reproducible path. Each search ran within its own time limit (Table 3 of the report).
Search databases (SQLite) should be on a local disk: on the network drive used here they could not lock files.

**What this repository is designed to rebuild on its own** (a clean install was tested for the system and the demos). Every reported number (`results/`, `tools/collect_results.py`), every
figure except the two image examples (they need the virtual-stain caches), the final Q2 models (`tools/fit_final_models.py`
refits them from `results/`), the system, the demos and the notebook. Refitting the final Q3 and Q4 models also needs the
per-cell tables and the chip-image embeddings (2 GB, not in git): rerun `sweep/q3_search.py features` and
`sweep/q4_chip_qc.py embed` to recreate them.

## Data, licences, AI disclosure
CPJUMP1 (Cell Painting Gallery cpg0000, CC0) · OoC Image Dataset (Zenodo 10203721, CC BY 4.0, doi:10.3390/data9020028, used for
evaluation; 8 example images redistributed as centre crops in `apps/web_demo/examples`, with attribution) · code MIT. The code, analyses, figures, report and video were built with Claude (Anthropic, via Claude Code), following the team's specifications and decisions, and were reviewed by tiagomgfernandes, who asked for adjustments in later prompts and also edited them directly.
