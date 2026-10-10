"""Train the bright-field -> Cell Painting U-Net on train-split compounds (+ DMSO) of the train plate(s).

    python legacy/train.py --run base --steps 8000
"""
import os as _os, sys as _sys  # legacy script: shared code from the repository root, runs from legacy/runs
from pathlib import Path as _P
_sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
_os.environ.setdefault("VCP_RUNS", str(_P(__file__).resolve().parent / "runs"))
import argparse
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

import config as C
from data import CropDataset, input_spec, load_plate, norm_params, normalise, predict
from model import PatchDiscriminator, UNet


def _gauss(ch, size=11, sigma=1.5, dev="cuda"):
    g = torch.exp(-(torch.arange(size, device=dev) - size // 2) ** 2 / (2 * sigma ** 2))
    g = (g / g.sum())[:, None] * (g / g.sum())[None, :]
    return g.expand(ch, 1, size, size)


def ssim_loss(x, y):
    """1 - SSIM, per channel, Gaussian window. Inputs are roughly in [0, 1]."""
    ch = x.shape[1]
    w = _gauss(ch, dev=x.device)
    f = lambda t: F.conv2d(t, w, groups=ch)
    mx, my = f(x), f(y)
    vx, vy, cxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mx * my + c1) * (2 * cxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (vx + vy + c2))
    return 1 - s.mean()


def grad_loss(x, y):
    """L1 on finite differences: penalises missing edges and fine texture."""
    dx = lambda t: t[..., :, 1:] - t[..., :, :-1]
    dy = lambda t: t[..., 1:, :] - t[..., :-1, :]
    return F.l1_loss(dx(x), dx(y)) + F.l1_loss(dy(x), dy(y))


def fft_loss(x, y):
    """L1 between log-amplitude spectra: an L1/L2-trained net under-predicts high-frequency power."""
    fx, fy = torch.fft.rfft2(x, norm="ortho").abs(), torch.fft.rfft2(y, norm="ortho").abs()
    return F.l1_loss(torch.log1p(fx), torch.log1p(fy))


@torch.no_grad()
def validate(model, cfg, dev, max_sites=48):
    """Per-channel Pearson r on whole fields from held-out val-split compounds of the train plate(s)."""
    model.eval()
    rs = []
    for p in C.TRAIN_PLATES:
        arr, idx = load_plate(p)
        lo, hi = norm_params(p)
        sel = np.where(idx.split == "val")[0][:max_sites]
        for i in sel:
            x = normalise(np.asarray(arr[i]), lo, hi)
            pred = predict(model, cfg, x, dev)
            tgt = x[C.TARGET_CH]
            rs.append([np.corrcoef(pred[c].ravel(), tgt[c].ravel())[0, 1] for c in range(len(C.TARGET_CH))])
    model.train()
    return np.nanmean(rs, 0) if rs else np.full(len(C.TARGET_CH), np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="base")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--bs", type=int, default=12)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--base", type=int, default=32)
    ap.add_argument("--val-every", type=int, default=1000)
    ap.add_argument("--w-ssim", type=float, default=0.0)
    ap.add_argument("--w-grad", type=float, default=0.0)
    ap.add_argument("--w-fft", type=float, default=0.0)
    ap.add_argument("--inputs", nargs="+", default=None, help="bright-field planes, e.g. Brightfield (default: all 3)")
    ap.add_argument("--bf-norm", choices=["plate", "image"], default="plate")
    ap.add_argument("--train-plates", nargs="+", default=C.TRAIN_PLATES)
    ap.add_argument("--init", default=None, help="start from runs/<init>/best.pt (fine-tuning)")
    ap.add_argument("--w-gan", type=float, default=0.0, help="adversarial (PatchGAN, hinge) weight")
    ap.add_argument("--select", choices=["r", "last"], default="r",
                    help="checkpoint to save as best.pt: highest val Pearson r, or the last one (for GAN runs, "
                         "where r favours the blurrier early checkpoints)")
    ap.add_argument("--aug", action="store_true", help="scale / blur / gamma / noise jitter for cross-microscope use")
    args = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    out = C.RUNS / args.run
    out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"

    cfg = vars(args)
    inputs, _ = input_spec(cfg)
    ds = CropDataset(args.train_plates, ["train", "control"], crop=args.crop, samples_per_epoch=args.steps * args.bs,
                     inputs=inputs, bf_norm=args.bf_norm, aug=args.aug)
    print(f"training sites in RAM: {len(ds.x)}", flush=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.bs, num_workers=6, pin_memory=True, drop_last=True,
                                     worker_init_fn=lambda w: np.random.seed(torch.initial_seed() % 2**32))

    model = UNet(len(inputs), len(C.TARGET_CH), base=args.base).to(dev)
    if args.init:
        model.load_state_dict(torch.load(C.RUNS / args.init / "best.pt", map_location=dev))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)
    if args.w_gan:
        disc = PatchDiscriminator(len(inputs) + len(C.TARGET_CH)).to(dev)
        opt_d = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

    log, best, t0 = [], -1.0, time.time()
    for step, (x, y) in enumerate(dl, 1):
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x)
        pred = pred.float()
        parts = {"l1": F.l1_loss(pred, y)}
        if args.w_ssim:
            parts["ssim"] = args.w_ssim * ssim_loss(pred, y)
        if args.w_grad:
            parts["grad"] = args.w_grad * grad_loss(pred, y)
        if args.w_fft:
            parts["fft"] = args.w_fft * fft_loss(pred, y)
        if args.w_gan:
            # discriminator step (hinge), then the generator's adversarial term
            d_loss = F.relu(1 - disc(x, y)).mean() + F.relu(1 + disc(x, pred.detach())).mean()
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
            parts["gan"] = args.w_gan * -disc(x, pred).mean()
        loss = sum(parts.values())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} [" + " ".join(f"{k}={v.item():.4f}" for k, v in parts.items())
                  + f"] ({time.time() - t0:.0f}s)", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            r = validate(model, cfg, dev)
            log.append({"step": step, "loss": loss.item(), **{f"r_{C.CHANNELS[c]}": float(v) for c, v in zip(C.TARGET_CH, r)}})
            print("  val r:", " ".join(f"{C.CHANNELS[c]}={v:.3f}" for c, v in zip(C.TARGET_CH, r)), flush=True)
            torch.save(model.state_dict(), out / "last.pt")
            score = float(np.nanmean(r)) if np.isfinite(r).any() else 0.0  # no val sites -> keep latest
            if score >= best or args.select == "last":
                best = score
                torch.save(model.state_dict(), out / "best.pt")
        if step >= args.steps:
            break
    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "log": log, "best_mean_r": best}, indent=1))


if __name__ == "__main__":
    main()
