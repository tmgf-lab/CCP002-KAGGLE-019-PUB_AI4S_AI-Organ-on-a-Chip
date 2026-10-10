"""Loading packed plates, normalisation, and the training crop dataset."""
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage as ndi

import config as C


def data_file(name):
    """First existing C.DATA_DIRS/name (plates may live in different folders)."""
    for d in C.DATA_DIRS:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"{name} not found in {[str(d) for d in C.DATA_DIRS]}")


def load_plate(plate):
    """Returns (memmap array, index DataFrame joined with the plate map)."""
    arr = np.load(data_file(f"{plate}.npy"), mmap_mode="r")
    idx = pd.read_csv(data_file(f"{plate}_index.csv"))
    wells = pd.read_csv(data_file("wells.csv"))
    idx = idx.merge(wells, on="well", how="left")
    return arr, idx


def norm_params(plate):
    for d in C.DATA_DIRS:
        f = d / "norm.json"
        if f.exists() and plate in (n := json.loads(f.read_text())):
            lo = np.array(n[plate]["lo"], np.float32)[:, None, None]
            hi = np.array(n[plate]["hi"], np.float32)[:, None, None]
            return lo, hi
    raise KeyError(f"no normalisation for {plate}")


def normalise(x, lo, hi):
    """uint16 (C,H,W) -> float32 roughly in [0, 1]; values above the plate's 99.9th pct are kept (not clipped)."""
    return (x.astype(np.float32) - lo) / (hi - lo)


def input_spec(cfg):
    """Input channel indices and bright-field normalisation mode of a trained run (defaults = original model)."""
    names = cfg.get("inputs") or [C.CHANNELS[c] for c in C.INPUT_CH]
    return [C.CHANNELS.index(n) for n in names], cfg.get("bf_norm", "plate")


def prep_input(bf, bf_norm):
    """bf: (k, H, W) plate-normalised bright-field. 'image' mode rescales each image to its own 1-99th
    percentiles, which makes the model usable on other microscopes (e.g. organ-on-a-chip images)."""
    if bf_norm == "image":
        lo = np.percentile(bf, 1, axis=(1, 2), keepdims=True)
        hi = np.percentile(bf, 99, axis=(1, 2), keepdims=True)
        bf = (bf - lo) / (hi - lo + 1e-6)
    return np.ascontiguousarray(bf, dtype=np.float32)


LOG_K = 10.0  # log target: y' = log1p(K*max(y,0)) / log1p(K); compresses bright peaks


def to_target_space(y, transform):
    return np.log1p(LOG_K * np.clip(y, 0, None)) / np.log1p(LOG_K) if transform == "log" else y


def from_target_space(y, transform):
    return np.expm1(y * np.log1p(LOG_K)) / LOG_K if transform == "log" else y


def load_model(run_dir, ckpt="best.pt", dev="cuda"):
    from model import build_model
    cfg = json.loads((run_dir / "train_log.json").read_text())["args"]
    inputs, _ = input_spec(cfg)
    model = build_model(cfg, len(inputs), len(C.TARGET_CH), pretrained=False).to(dev).eval()
    model.load_state_dict(torch.load(run_dir / ckpt, map_location=dev))
    return model, cfg


@torch.no_grad()
def predict(model, cfg, x, dev="cuda"):
    """x: normalised (8, H, W) field (or (k, H, W) bright-field only if it matches the run's inputs)."""
    inputs, bf_norm = input_spec(cfg)
    bf = x[inputs] if x.shape[0] == len(C.CHANNELS) else x
    inp = torch.from_numpy(prep_input(bf, bf_norm))[None].to(dev)
    dtype = torch.bfloat16 if dev == "cuda" and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    with torch.autocast("cuda", dtype=dtype, enabled=dev == "cuda"):
        out = model(inp).float()[0].cpu().numpy()
    return from_target_space(out, cfg.get("target_transform", "linear"))


class CropDataset(torch.utils.data.Dataset):
    """Random crops from the sites of the given splits, held in RAM as float16 after normalisation."""

    def __init__(self, plates, splits, crop=256, samples_per_epoch=4000, flips=True,
                 inputs=None, bf_norm="plate", aug=False, target_transform="linear", fraction=1.0, seed=0):
        parts = []
        for p in plates:
            arr, idx = load_plate(p)
            parts.append((arr, np.where(idx.split.isin(splits))[0], *norm_params(p)))
        n = sum(len(sel) for _, sel, _, _ in parts)
        shape = parts[0][0].shape[1:]
        self.x = np.empty((n, *shape), np.float16)  # filled in place: no list + stack double copy
        k = 0
        for arr, sel, lo, hi in parts:
            for i in sel:
                self.x[k] = normalise(np.asarray(arr[i]), lo, hi)
                k += 1
        if fraction < 1.0:  # a fixed random subset, nested: the 25% subset is inside the 50% subset
            order = np.random.default_rng(seed).permutation(len(self.x))
            self.x = self.x[np.sort(order[:max(1, int(round(fraction * len(self.x))))])]
        self.crop, self.n, self.flips = crop, samples_per_epoch, flips
        self.inputs, self.bf_norm, self.aug, self.tt = inputs or C.INPUT_CH, bf_norm, aug, target_transform

    def __len__(self):
        return self.n

    def _n_fields(self):
        return len(self.x)

    def _crop(self, j, y0, x0, size):
        return self.x[j][:, y0:y0 + size, x0:x0 + size].astype(np.float32)

    def __getitem__(self, _):
        pool = getattr(self, "idx", None)
        j = pool[np.random.randint(len(pool))] if pool is not None else np.random.randint(self._n_fields())
        H = 540
        # Scale jitter (+-30%) so the model tolerates other magnifications / pixel sizes.
        size = min(H, int(round(self.crop * np.random.uniform(0.75, 1.33)))) if self.aug else self.crop
        y0, x0 = np.random.randint(0, H - size + 1, 2)
        c = self._crop(j, y0, x0, size)
        if size != self.crop:
            c = F.interpolate(torch.from_numpy(c)[None], size=(self.crop, self.crop), mode="bilinear",
                              antialias=True, align_corners=False)[0].numpy()
        if self.flips:
            if np.random.rand() < 0.5:
                c = c[:, :, ::-1]
            if np.random.rand() < 0.5:
                c = c[:, ::-1]
            c = np.rot90(c, np.random.randint(4), axes=(1, 2))
        c = np.ascontiguousarray(c)
        inp = prep_input(c[self.inputs], self.bf_norm)
        if self.aug:  # photometric jitter on the bright-field input only
            if np.random.rand() < 0.3:
                inp = ndi.gaussian_filter(inp, (0, *(np.random.uniform(0.3, 1.2),) * 2))
            inp = np.clip(inp, 0, 2) ** np.random.uniform(0.7, 1.4)
            inp = inp + np.random.normal(0, np.random.uniform(0, 0.03), inp.shape)
            inp = inp.astype(np.float32)
        tgt = to_target_space(c[C.TARGET_CH], self.tt).astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(inp)), torch.from_numpy(np.ascontiguousarray(tgt))


class LazyCropDataset(CropDataset):
    """Same crops as CropDataset, but read from the packed uint16 files on demand (no RAM copy).
    The OS page cache is shared between processes, so several trainers can use the same plates."""

    def __init__(self, plates, splits, crop=256, inputs=None, bf_norm="plate", aug=False, target_transform="linear"):
        self.entries = []
        for p in plates:
            arr, idx = load_plate(p)
            lo, hi = norm_params(p)
            self.entries += [(arr, int(i), lo, hi) for i in np.where(idx.split.isin(splits))[0]]
        self.crop, self.n, self.flips, self.idx = crop, 1, True, None
        self.inputs, self.bf_norm, self.aug, self.tt = inputs or C.INPUT_CH, bf_norm, aug, target_transform

    def _n_fields(self):
        return len(self.entries)

    def _crop(self, j, y0, x0, size):
        arr, i, lo, hi = self.entries[j]
        return normalise(np.asarray(arr[i, :, y0:y0 + size, x0:x0 + size]), lo, hi)


@torch.no_grad()
def predict_multi(models, cfg, x, dev="cuda", tta=False):
    """Average of predict() over several models (seed ensemble) and, if tta, over the 8 flips/rotations."""
    outs = []
    for m in models:
        if not tta:
            outs.append(predict(m, cfg, x, dev))
            continue
        for k in range(4):
            for flip in (False, True):
                xx = np.rot90(x, k, axes=(1, 2))
                xx = xx[:, :, ::-1] if flip else xx
                yy = predict(m, cfg, np.ascontiguousarray(xx), dev)
                yy = yy[:, :, ::-1] if flip else yy
                outs.append(np.rot90(yy, -k, axes=(1, 2)))
    return np.mean(outs, 0)
