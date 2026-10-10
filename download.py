"""Download a subset of CPJUMP1 (cpg0000, Cell Painting Gallery, CC0) and pack each plate into one array.

Output per plate: DATA/<plate>.npy, uint16 (n_sites, 8, 540, 540), plus DATA/<plate>_index.csv.
Also writes DATA/wells.csv (plate map + compound annotations + compound split) and DATA/norm.json.

    python download.py            # plates/sites from config.py
    python download.py --smoke    # 8 wells per plate, for a quick end-to-end check
"""
import argparse
import hashlib
import io
import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import tifffile

import config as C

URL_COLS = {"DNA": "URL_OrigDNA", "ER": "URL_OrigER", "RNA": "URL_OrigRNA", "AGP": "URL_OrigAGP",
            "Mito": "URL_OrigMito", "LowZBF": "URL_OrigLowZBF", "Brightfield": "URL_OrigBrightfield",
            "HighZBF": "URL_OrigHighZBF"}


def fetch(url, tries=8):
    for i in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(min(2 ** i, 60))


def s3_to_https(u):
    return u.replace("s3://cellpainting-gallery", C.S3)


def build_wells():
    base = f"{C.S3}/{C.PILOT}/metadata"
    pm = pd.read_csv(io.BytesIO(fetch(f"{base}/platemaps/{C.BATCH}/platemap/JUMP-Target-1_compound_platemap.txt")), sep="\t")
    meta = pd.read_csv(io.BytesIO(fetch(f"{base}/external_metadata/JUMP-Target-1_compound_metadata_targets.tsv")), sep="\t")
    w = pm.merge(meta, on="broad_sample", how="left")
    w["pert_iname"] = w["pert_iname"].fillna("DMSO")
    w.loc[w.broad_sample.isna(), "control_type"] = "negcon"
    # Deterministic compound split by hash, so it never depends on which plates were downloaded.
    def split(name):
        if name == "DMSO":
            return "control"
        h = int(hashlib.md5(f"{C.SPLIT_SEED}:{name}".encode()).hexdigest(), 16) % 1000 / 1000
        return "train" if h < C.SPLIT_FRAC["train"] else "val" if h < C.SPLIT_FRAC["train"] + C.SPLIT_FRAC["val"] else "test"
    w["split"] = w.pert_iname.map(split)
    w = w.rename(columns={"well_position": "well"})
    return w


def load_site(row):
    chans = []
    for ch in C.CHANNELS:
        img = tifffile.imread(io.BytesIO(fetch(s3_to_https(row[URL_COLS[ch]])))).astype(np.float32)
        f = C.DOWNSAMPLE
        img = img.reshape(img.shape[0] // f, f, img.shape[1] // f, f).mean((1, 3))
        chans.append(img)
    return np.stack(chans).round().clip(0, 65535).astype(np.uint16)


def download_plate(plate, sites, wells_keep, workers):
    ld = pd.read_csv(io.BytesIO(fetch(f"{C.S3}/{C.PILOT}/load_data_csv/{C.BATCH}/{plate}/load_data.csv")))
    missing = sorted(set(sites) - set(ld.Metadata_Site))
    if missing:  # e.g. U2OS plates have 9 sites per well, A549 plates 16
        raise SystemExit(f"{plate}: sites {missing} do not exist (plate has {sorted(ld.Metadata_Site.unique())})")
    ld = ld[ld.Metadata_Site.isin(sites) & ld.Metadata_Well.isin(wells_keep)]
    ld = ld.sort_values(["Metadata_Well", "Metadata_Site"]).reset_index(drop=True)
    out = C.DATA / f"{plate}.npy"
    done_path = C.DATA / f"{plate}_done.npy"
    idx_path = C.DATA / f"{plate}_index.csv"
    side = 1080 // C.DOWNSAMPLE
    shape = (len(ld), len(C.CHANNELS), side, side)
    # Resumable: reuse the packed array if it has the same layout, and skip sites already marked done.
    resume = out.exists() and done_path.exists() and np.load(out, mmap_mode="r").shape == shape
    arr = np.lib.format.open_memmap(out, mode="r+" if resume else "w+", dtype=np.uint16, shape=shape)
    done = np.load(done_path) if resume else np.zeros(len(ld), bool)
    todo = np.where(~done)[0]
    print(f"  {plate}: {done.sum()} sites already done, {len(todo)} to fetch", flush=True)

    def job(i):
        try:
            arr[i] = load_site(ld.iloc[i])
            return i, True
        except Exception as e:
            print(f"  site {i} failed: {e}", flush=True)
            return i, False

    with ThreadPoolExecutor(workers) as ex:
        for k, (i, ok) in enumerate(ex.map(job, todo), 1):
            done[i] = ok
            if k % 50 == 0 or k == len(todo):
                arr.flush()
                np.save(done_path, done)
                print(f"  {plate}: {done.sum()}/{len(ld)} sites", flush=True)
    arr.flush()
    np.save(done_path, done)
    if not done.all():
        raise SystemExit(f"{plate}: {(~done).sum()} sites failed; rerun download.py to resume")
    ld[["Metadata_Plate", "Metadata_Well", "Metadata_Site"]].rename(
        columns=lambda c: c.replace("Metadata_", "").lower()).to_csv(idx_path, index=False)
    return out


def plate_norm(plate, n=64):
    """Per-plate, per-channel 0.5/99.9 percentiles. Plate-level (not per-image) scaling keeps
    well-to-well intensity and density differences, which carry the phenotype."""
    a = np.load(C.DATA / f"{plate}.npy", mmap_mode="r")
    sel = np.linspace(0, len(a) - 1, min(n, len(a))).astype(int)
    x = np.asarray(a[sel], dtype=np.float32)[:, :, ::4, ::4]
    lo = np.percentile(x, 0.5, axis=(0, 2, 3))
    hi = np.percentile(x, 99.9, axis=(0, 2, 3))
    return {"lo": lo.tolist(), "hi": hi.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--plates", nargs="+", help="extra evaluation plates (default: config train + test plates)")
    ap.add_argument("--sites", nargs="+", type=int, default=None,
                    help="sites per well (default: config SITES_TRAIN for training plates, SITES_TEST otherwise)")
    ap.add_argument("--sample", type=int, default=0, help="only the first N wells of each split (quick demos)")
    args = ap.parse_args()
    C.DATA.mkdir(parents=True, exist_ok=True)

    wells = build_wells()
    wells.to_csv(C.DATA / "wells.csv", index=False)
    print(wells.split.value_counts().to_string())

    keep = set(wells.well)
    if args.sample:
        keep = set(wells.groupby("split").head(args.sample).well)
    elif args.smoke:
        keep = set(wells[wells.split == "control"].well[:2]) | set(wells[wells.split == "train"].well[:3]) \
            | set(wells[wells.split == "val"].well[:1]) \
            | set(wells[wells.split == "test"].well[:2])

    norm_path = C.DATA / "norm.json"
    norm = json.loads(norm_path.read_text()) if norm_path.exists() else {}
    for plate in args.plates or C.TRAIN_PLATES + C.TEST_PLATES:
        sites = args.sites or (C.SITES_TRAIN if plate in C.TRAIN_PLATES else C.SITES_TEST)
        print(f"{plate}: sites {sites}, {len(keep)} wells", flush=True)
        download_plate(plate, sites, keep, args.workers)
        norm[plate] = plate_norm(plate)
        norm_path.write_text(json.dumps(norm, indent=1))


if __name__ == "__main__":
    main()
