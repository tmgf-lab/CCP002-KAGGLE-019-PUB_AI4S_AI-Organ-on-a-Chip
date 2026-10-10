"""Q4 building blocks: data index (folder or ZIP), preprocessing, hand-crafted / texture features, frozen
embeddings (timm, open_clip) with a resumable chunked cache, and the throughput benchmark.

Self-contained (numpy, pandas, PIL, scipy, scikit-image, torch, timm, open_clip) so it runs on Kaggle with
only the `sweep/` package uploaded.

Embedding directory layout (`--emb-dir`):
    index.csv                       imageID, experiment, cell, day, good (row order of every array)
    meta.json                       resolutions, model names, smoke flag, timings
    {representation}__{prep}.npy    float32 (n_images, d)
    pix__{prep}.npy                 uint8 (n_images, T, 224, 299) grey pixels for fine-tuning / maps (T=4 for tiles)
    parts/                          per-chunk pieces while a run is in progress (resume = skip existing pieces)
"""
import io
import json
import os
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
PREPS = ["centre_crop", "resize", "tiles_mean", "clahe+centre_crop"]
CROP = (1024, 768)        # (W, H) native-pixel centre crop = one quadrant of the 2048 x 1536 image
LO = (224, 299)           # (H, W) CNN input, also the fine-tuning pixels
FEAT_HW = (384, 512)      # (H, W) for hand-crafted / texture features
# representation -> (library, model name, (H, W) input)
DEEP = {
    "resnet50": ("timm", "resnet50.a1_in1k", LO),
    "efficientnet_b3": ("timm", "efficientnet_b3.ra2_in1k", (288, 384)),
    "convnext_tiny": ("timm", "convnext_tiny.fb_in1k", LO),
    "dinov2_small": ("timm", "vit_small_patch14_dinov2.lvd142m", (448, 588)),
    "dinov2_base": ("timm", "vit_base_patch14_dinov2.lvd142m", (448, 588)),
    "dinov2_large": ("timm", "vit_large_patch14_dinov2.lvd142m", (448, 588)),
    "clip_vit_b16": ("open_clip", "ViT-B-16/openai", (224, 224)),
}
CPU_REPS = ["handcrafted", "glcm_lbp"]
ALL_REPS = CPU_REPS + list(DEEP) + ["virtual_stain_features"]


def safe(name):
    return name.replace("+", "_plus_")


def emb_file(d, rep, prep):
    return Path(d) / f"{rep}__{safe(prep)}.npy"


# ---------------------------------------------------------------------------------------------- index
def _parse(rel):
    parts = [q.lower() for q in Path(rel).parts]
    return {"imageID": Path(rel).stem,
            "zsplit": next((q for q in parts if q in ("train", "val", "test")), None),
            "label_dir": next((q for q in parts if q in ("good", "bad")), None)}


def build_index(xlsx, zip_path=None, img_dir=None):
    """One row per image: src (file path or 'zip:member'), imageID, cell, day, good, experiment.
    Same rules as chip_eval.index_images: label from the datasheet (1 = good), else from the folder."""
    rows = []
    if zip_path:
        with zipfile.ZipFile(zip_path) as z:
            for m in z.namelist():
                if Path(m).suffix.lower() in IMG_EXT and not Path(m).name.startswith("._") and "__MACOSX" not in m:
                    rows.append({"src": "zip:" + m, **_parse(m)})
    else:
        for p in Path(img_dir).rglob("*"):
            if p.suffix.lower() in IMG_EXT:
                rows.append({"src": str(p), **_parse(p.relative_to(img_dir))})
    idx = pd.DataFrame(rows)
    sheet = pd.read_excel(xlsx).rename(columns={"cell type": "cell", "Decision 1/2 (good/bad)": "decision"})
    sheet["imageID"] = sheet.imageID.astype(str)
    idx = idx.merge(sheet[["imageID", "cell", "day", "decision"]], on="imageID", how="left")
    idx["good"] = np.where(idx.decision.notna(), idx.decision == 1, idx.label_dir == "good").astype(int)
    idx["experiment"] = idx.imageID.str.extract(r"^(\d+)", expand=False).fillna(idx.imageID)
    idx["cell"] = idx.cell.fillna("unknown").astype(str)
    return idx.sort_values("imageID").reset_index(drop=True)


_ZIP = {}


def read_grey(src, zip_path=None):
    if src.startswith("zip:"):
        z = _ZIP.get(zip_path)
        if z is None:
            z = _ZIP[zip_path] = zipfile.ZipFile(zip_path)
        return Image.open(io.BytesIO(z.read(src[4:]))).convert("L")
    return Image.open(src).convert("L")


# ---------------------------------------------------------------------------------------------- preprocessing
def _crop(im, l, t):
    return im.crop((l, t, l + CROP[0], t + CROP[1]))


def variants(im):
    """prep -> list of PIL grey images at native resolution (tiles_mean: 2 x 2 grid of quadrants)."""
    from skimage.exposure import equalize_adapthist
    W, H = im.size
    l, t = (W - CROP[0]) // 2, (H - CROP[1]) // 2
    centre = _crop(im, l, t)
    xs = [0, max(0, W - CROP[0])]
    ys = [0, max(0, H - CROP[1])]
    tiles = [_crop(im, x, y) for y in ys for x in xs]
    cl = equalize_adapthist(np.asarray(centre, np.float32) / 255.0, clip_limit=0.01)
    return {"centre_crop": [centre], "resize": [im], "tiles_mean": tiles,
            "clahe+centre_crop": [Image.fromarray((cl * 255).astype(np.uint8))]}


def channel_features(img):
    """Hand-crafted features (same recipe as evaluate.channel_features) on a 1-99 percentile normalised field."""
    from scipy import ndimage as ndi
    from skimage.filters import threshold_otsu
    lo, hi = np.percentile(img, [1, 99])
    img = (img - lo) / (hi - lo + 1e-6)
    f = {}
    q = np.percentile(img, [10, 50, 90, 99])
    f.update(mean=img.mean(), std=img.std(), p10=q[0], p50=q[1], p90=q[2], p99=q[3])
    try:
        f["frac_fg"] = (img > threshold_otsu(img)).mean()
    except ValueError:
        f["frac_fg"] = 0.0
    gy, gx = np.gradient(img)
    f["grad"] = np.hypot(gx, gy).mean()
    f["lap"] = np.abs(ndi.laplace(img)).mean()
    prev = img
    for s in (1, 2, 4, 8):
        g = ndi.gaussian_filter(img, s)
        f[f"dog{s}"] = np.abs(prev - g).mean()
        prev = g
    return np.array(list(f.values()), np.float32)


def glcm_lbp(img_u8):
    """GLCM (32 grey levels; distances 1, 2, 4; mean and range over 4 angles; 6 properties) + uniform LBP
    histograms (P=8, R=1 and P=16, R=2) -> 64 features."""
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    q = (img_u8 // 8).astype(np.uint8)
    m = graycomatrix(q, [1, 2, 4], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=32, symmetric=True, normed=True)
    out = []
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"):
        v = np.nan_to_num(graycoprops(m, prop))   # (distances, angles)
        out += list(v.mean(1)) + list(v.max(1) - v.min(1))
    for P, R in ((8, 1), (16, 2)):
        lbp = local_binary_pattern(img_u8, P, R, method="uniform")
        out += list(np.bincount(lbp.astype(int).ravel(), minlength=P + 2)[: P + 2] / lbp.size)
    return np.array(out, np.float32)


def decode(args):
    """One image -> {'hc': {prep: vec}, 'glcm': {prep: vec}, 'px': {(prep, (H, W)): uint8 (T, H, W)}}."""
    src, zip_path, sizes, cpu_reps = args
    var = variants(read_grey(src, zip_path))
    out = {"hc": {}, "glcm": {}, "px": {}}
    for p, ims in var.items():
        if cpu_reps:
            small = [np.asarray(im.resize(FEAT_HW[::-1], Image.BILINEAR)) for im in ims]
            if "handcrafted" in cpu_reps:
                out["hc"][p] = np.mean([channel_features(s.astype(np.float32)) for s in small], 0)
            if "glcm_lbp" in cpu_reps:
                out["glcm"][p] = np.mean([glcm_lbp(s) for s in small], 0)
        for hw in sizes:
            out["px"][(p, hw)] = np.stack([np.asarray(im.resize(hw[::-1], Image.BILINEAR), np.uint8) for im in ims])
    return out


# ---------------------------------------------------------------------------------------------- deep models
class Embedder:
    def __init__(self, rep, device):
        import torch
        lib, name, self.hw = DEEP[rep]
        self.rep, self.device = rep, device
        if lib == "timm":
            import timm
            kw = {"dynamic_img_size": True} if "dinov2" in name else {}
            self.model = timm.create_model(name, pretrained=True, num_classes=0, **kw).eval()
            cfg = timm.data.resolve_data_config({}, model=self.model)
            mean, std = cfg["mean"], cfg["std"]
            self.fwd = self.model
        else:
            import open_clip
            arch, tag = name.split("/")
            self.model, _, _ = open_clip.create_model_and_transforms(arch, pretrained=tag)
            self.model.eval()
            mean, std = open_clip.OPENAI_DATASET_MEAN, open_clip.OPENAI_DATASET_STD
            self.fwd = self.model.encode_image
        self.half = device.startswith("cuda")
        self.model.to(device, torch.float16 if self.half else torch.float32)
        self.mean = torch.tensor(mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, 3, 1, 1)

    def __call__(self, px, bs=32):
        """px: uint8 (N, H, W) -> float32 (N, d)."""
        import torch
        out = []
        with torch.no_grad():
            for i in range(0, len(px), bs):
                x = torch.from_numpy(np.ascontiguousarray(px[i:i + bs])).to(self.device).float().div_(255)[:, None]
                x = ((x.expand(-1, 3, -1, -1) - self.mean) / self.std)
                x = x.half() if self.half else x
                out.append(self.fwd(x).float().cpu().numpy())
        return np.concatenate(out)


def flops_per_image(rep):
    """Forward FLOPs of one input (torch FlopCounterMode; matmul/conv only)."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    e = Embedder(rep, "cpu")
    x = torch.zeros(1, 3, *e.hw)
    with FlopCounterMode(display=False) as fc, torch.no_grad():
        e.fwd(x)
    return fc.get_total_flops()


# ---------------------------------------------------------------------------------------------- embed driver
def _todo(emb_dir, reps, preps, k):
    parts = Path(emb_dir) / "parts"
    need = [(r, p) for r in reps for p in preps if not emb_file(emb_dir, r, p).exists()
            and not (parts / f"{r}__{safe(p)}__c{k:04d}.npy").exists()]
    need += [("pix", p) for p in preps if not emb_file(emb_dir, "pix", p).exists()
             and not (parts / f"pix__{safe(p)}__c{k:04d}.npy").exists()]
    return need


def embed_shard(emb_dir, srcs, zip_path, reps, preps, device, workers, chunk, shard=0, n_shards=1, log=print):
    """Process chunks k with k % n_shards == shard; every (rep, prep) piece of a chunk is saved on its own."""
    from multiprocessing import get_context
    import torch
    if device == "cpu":
        torch.set_num_threads(max(1, workers))
    parts = Path(emb_dir) / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    deep = [r for r in reps if r in DEEP]
    models = {}
    n_chunks = (len(srcs) + chunk - 1) // chunk
    with get_context("spawn").Pool(workers) as pool:
        for k in range(shard, n_chunks, n_shards):
            need = _todo(emb_dir, reps, preps, k)
            if not need:
                continue
            t0 = time.time()
            need_reps = {r for r, _ in need}
            sizes = sorted({DEEP[r][2] for r in deep if r in need_reps} | ({LO} if "pix" in need_reps else set()))
            cpu = [r for r in CPU_REPS if r in need_reps]
            dec = pool.map(decode, [(s, zip_path, sizes, cpu) for s in srcs[k * chunk:(k + 1) * chunk]], chunksize=2)
            t1 = time.time()
            for r, p in need:
                f = parts / f"{r}__{safe(p)}__c{k:04d}.npy"
                if r == "handcrafted":
                    a = np.stack([d["hc"][p] for d in dec])
                elif r == "glcm_lbp":
                    a = np.stack([d["glcm"][p] for d in dec])
                elif r == "pix":
                    a = np.stack([d["px"][(p, LO)] for d in dec])
                else:
                    if r not in models:
                        models[r] = Embedder(r, device)
                    px = np.stack([d["px"][(p, DEEP[r][2])] for d in dec])   # (n, T, H, W)
                    n, T = px.shape[:2]
                    a = models[r](px.reshape(n * T, *px.shape[2:])).reshape(n, T, -1).mean(1)
                np.save(f, a.astype(np.uint8 if r == "pix" else np.float32))
            log(f"[shard {shard}] chunk {k + 1}/{n_chunks}: {len(need)} pieces, decode {t1 - t0:.0f}s, "
                f"models {time.time() - t1:.0f}s")


def merge(emb_dir, reps, preps, n):
    parts = Path(emb_dir) / "parts"
    chunks = None
    for r in list(reps) + ["pix"]:
        for p in preps:
            f = emb_file(emb_dir, r, p)
            if f.exists():
                continue
            pieces = sorted(parts.glob(f"{r}__{safe(p)}__c*.npy"))
            if not pieces:
                continue
            a = np.concatenate([np.load(q) for q in pieces])
            if len(a) != n:
                print(f"  {r}/{p}: {len(a)}/{n} rows so far (incomplete, not merged)")
                continue
            np.save(f, a)
            for q in pieces:
                q.unlink()
            chunks = len(pieces)
    return chunks


def virtual_features(emb_dir, idx, csv):
    """virtual_stain_features: the virt_* columns of chip_features.csv (computed on the centre crop only)."""
    f = emb_file(emb_dir, "virtual_stain_features", "centre_crop")
    if f.exists() or not csv or not Path(csv).exists():
        return f.exists()
    d = pd.read_csv(csv, dtype={"imageID": str})
    cols = [c for c in d.columns if c.startswith("virt_")]
    m = idx[["imageID"]].merge(d[["imageID"] + cols], on="imageID", how="left")
    if m[cols].isna().all(1).any():
        print(f"  virtual features missing for {int(m[cols].isna().all(1).sum())} images -> skipped")
        return False
    np.save(f, m[cols].to_numpy(np.float32))
    (Path(emb_dir) / "virtual_stain_features_columns.txt").write_text("\n".join(cols))
    return True


def benchmark(srcs, zip_path, reps, n_img, workers):
    """CPU seconds per image: decode + CPU features (all 4 preps) and each deep model (7 forwards / image:
    centre, resize, clahe, 4 tiles). FLOPs are counted to extrapolate to a T4."""
    import torch
    from multiprocessing import get_context
    torch.set_num_threads(workers)
    res = {}
    srcs = srcs[:n_img]
    deep = [r for r in reps if r in DEEP]
    sizes = sorted({DEEP[r][2] for r in deep} | {LO})
    with get_context("spawn").Pool(workers) as pool:
        pool.map(decode, [(srcs[0], zip_path, [LO], [])])   # warm the workers
        t0 = time.time()
        pool.map(decode, [(s, zip_path, [LO], []) for s in srcs], chunksize=1)
        res["decode_only_s_per_img"] = (time.time() - t0) / len(srcs)
        t0 = time.time()
        dec = pool.map(decode, [(s, zip_path, sizes, CPU_REPS) for s in srcs], chunksize=1)
        res["decode_all_s_per_img"] = (time.time() - t0) / len(srcs)
    for r in deep:
        e = Embedder(r, "cpu")
        px = np.concatenate([np.concatenate([d["px"][(p, DEEP[r][2])] for p in PREPS]) for d in dec])
        e(px[:2], bs=2)
        t0 = time.time()
        e(px, bs=8)
        dt = time.time() - t0
        res[r] = {"cpu_s_per_forward": dt / len(px), "cpu_s_per_image_7fwd": dt / len(px) * 7,
                  "gflops_per_forward": flops_per_image(r) / 1e9, "input_hw": DEEP[r][2]}
        del e
    return res
