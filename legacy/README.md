# Legacy: the starting solution (before the guided search)

The first version of the project (25–26 September 2026), kept for reference. The main results in the report come from the
guided searches (`sweep/`) and the Q1 winner (`runs/winner`); this folder holds what came before.

| Item | What it is | Where it is used now |
|---|---|---|
| `runs/base` | starting U-Net (width 32, transposed convolutions), trained by `train.py` | the Q1 baseline (validation r 0.955 vs 0.959 for the winner) |
| `runs/chip1` | single bright-field plane model, robustness-augmented | the web demo's "your own image" tab; the Q4 "virtual-stain features" option (`chip_s2/chip_features.csv`) |
| `runs/gan`, `runs/hf` | GAN and texture-loss variants | the "earlier exploratory experiments" paragraph of the report (§4) |
| `train.py` | trained `runs/base` | reproduce the baseline |
| `chip_eval.py` | ran the single-plane model on the chip images (`runs/chip1/chip_s2`) | reproduce the Q4 virtual-stain features |

The scripts use the code at the repository root and read/write `legacy/runs` (they set `VCP_RUNS` themselves):

```bash
python legacy/train.py --run base --steps 8000        # needs the A549 plates (python download.py)
python legacy/chip_eval.py --run chip1 --scale 1.0     # needs the organ-on-a-chip image dataset (Zenodo 10203721)
```
