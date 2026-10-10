"""Download the Q1 winner weights (U-Net width 64, bilinear; seeds 1 and 3; 121 MB each, too large for git).

    python weights.py            # -> runs/winner/seed1/best.pt, runs/winner/seed3/best.pt (checksums verified)

Source: the GitHub release "weights-v1" of this repository (public, no login).
"""
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RELEASE = "https://github.com/tmgf-lab/CCP002-KAGGLE-019-PUB_AI4S_AI-Organ-on-a-Chip/releases/download/weights-v1"
FILES = {"seed1": "9daaba12db881a75944ddd224e8e0c2ce53d45ed088cf80c33079ac2cd22876c",
         "seed3": "118e11755d85d064f4edde567baf385562ceed5f9fdbcef5b188439700fdb751"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def winner_dir():
    """runs/winner with the weights present (downloads them if needed)."""
    d = ROOT / "runs" / "winner"
    for seed, digest in FILES.items():
        out = d / seed / "best.pt"
        if out.exists() and sha256(out) == digest:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        ok = False
        if not RELEASE.startswith("["):
            url = f"{RELEASE}/winner_{seed}_best.pt"
            print("downloading", url, flush=True)
            try:
                urllib.request.urlretrieve(url, out)
                ok = sha256(out) == digest
            except Exception as e:  # noqa: BLE001
                print("  GitHub release not reachable:", e)
        if not ok:
            raise SystemExit(f"could not get {out}: download winner_{seed}_best.pt from the release weights-v1 (see README)")
    return d


def winner_json():
    """A q1_winner.json pointing at the local seed folders (same inference mode as chosen on validation)."""
    d = winner_dir()
    w = json.loads((d / "q1_winner_source.json").read_text())
    w["run_dirs"] = [str(d / Path(r).name.replace("t038_", "")) for r in w["run_dirs"]]
    out = d / "q1_winner.json"
    out.write_text(json.dumps(w, indent=1))
    return out


if __name__ == "__main__":
    print("winner:", winner_json())
