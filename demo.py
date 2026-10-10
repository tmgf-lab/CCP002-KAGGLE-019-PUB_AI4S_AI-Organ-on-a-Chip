"""Entry script. Stream one field straight from the Cell Painting Gallery, stain it virtually with the Q1 winner, count nuclei.
No data download needed (the winner weights are fetched once by weights.py). About 10 s on a CPU. Writes a side-by-side PNG.

Input : 3 bright-field z-planes of one CPJUMP1 field (streamed from the public S3 archive).
Output: 5 virtual Cell Painting stains (DNA, ER, RNA, AGP, Mito), nuclei counts (real vs virtual), pixel r per stain, a PNG.

    python demo.py --compound PP-121                 # a held-out cytotoxic compound
    python demo.py --compound DMSO --site 6
    python demo.py --plate BR00117019 --well M20 --out demo.png
"""
import argparse
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import config as C
from download import build_wells, fetch, load_site
from evaluate import nuclei
from data import load_model, predict, predict_multi
from weights import winner_dir

TGT = [C.CHANNELS[c] for c in C.TARGET_CH]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plate", default=C.TEST_PLATES[0], help="any plate listed in runs/<run>/norm.json")
    ap.add_argument("--compound", default="PP-121")
    ap.add_argument("--well", default=None, help="overrides --compound")
    ap.add_argument("--site", type=int, default=1)
    ap.add_argument("--run", default="winner", help="winner (Q1 search winner, 2 seeds) or a folder in runs/ (e.g. base)")
    ap.add_argument("--tta", action="store_true", help="test-time augmentation: the validated best mode, 8x slower")
    ap.add_argument("--out", default="demo.png")
    args = ap.parse_args()

    wells = build_wells()
    well = args.well or wells[wells.pert_iname == args.compound].well.iloc[0]
    info = wells[wells.well == well].iloc[0]
    ld = pd.read_csv(io.BytesIO(fetch(f"{C.S3}/{C.PILOT}/load_data_csv/{C.BATCH}/{args.plate}/load_data.csv")))
    row = ld[(ld.Metadata_Well == well) & (ld.Metadata_Site == args.site)].iloc[0]
    print(f"{args.plate} {well} site {args.site}: {info.pert_iname} (target {info.target if isinstance(info.target, str) else '-'}, split {info.split})")

    x = load_site(row).astype(np.float32)
    run = winner_dir() if args.run == "winner" else C.RUNS / args.run
    n = json.loads((run / "norm.json").read_text())[args.plate]
    lo, hi = np.array(n["lo"], np.float32)[:, None, None], np.array(n["hi"], np.float32)[:, None, None]
    x = (x - lo) / (hi - lo)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.run == "winner":
        ms = [load_model(run / s, dev=dev) for s in ("seed1", "seed3")]
        virt = predict_multi([m for m, _ in ms], ms[0][1], x, dev, args.tta)
    else:
        model, cfg = load_model(run, dev=dev)
        virt = predict(model, cfg, x, dev)
    real = x[C.TARGET_CH]
    n_real, _ = nuclei(real[0])
    n_virt, _ = nuclei(virt[0])
    print(f"nuclei: real stain {n_real}, virtual stain {n_virt}")
    print("pixel r (virtual vs real): " + "  ".join(f"{TGT[c]} {np.corrcoef(real[c].ravel(), virt[c].ravel())[0, 1]:.3f}" for c in range(5)))

    fig, axes = plt.subplots(2, 6, figsize=(15, 5.9))
    for r, (kind, imgs) in enumerate([("Real", real), ("Virtual", virt)]):
        for c, ax in enumerate(axes[r]):
            if c == 0 and r == 1:
                ax.axis("off")  # the bright-field input is shown once, top-left
                continue
            if c == 0:
                img = x[C.INPUT_CH[1]]
                vmin, vmax = np.percentile(img, [1, 99.5])
            else:
                img = imgs[c - 1]
                vmin, vmax = np.percentile(real[c - 1], [1, 99.7])
            ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(("Bright-field input" if c == 0 else f"{kind} {TGT[c - 1]}") if (r == 0 or c > 0) else "", fontsize=10)
    fig.suptitle(f"{info.pert_iname} ({args.plate} {well} s{args.site}) - nuclei: real {n_real}, virtual {n_virt}", fontsize=12)
    fig.tight_layout(h_pad=1.8)
    fig.savefig(args.out, dpi=130)
    print("saved", args.out)


if __name__ == "__main__":
    main()
