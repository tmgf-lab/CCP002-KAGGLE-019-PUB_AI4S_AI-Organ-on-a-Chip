"""End-to-end system: bright-field images in, a report with all four answers out.

    microscope bright-field fields ─► Q1 virtual stains ─┬─► Q2 per well: toxic? viability, drug effect vs DMSO
                                                       └─► Q3 per field: cells, counts, cell-state proportions
    organ-on-a-chip photos ────────► Q4 chip quality (P(good))          (separate input, same report)

Models: Q1 = the search winner (runs/winner, downloaded by weights.py); Q2/Q3/Q4 = models/ (tools/fit_final_models.py:
the winning paths fitted on calibration data). Q2 and Q3 were fitted on plate BR00117017: run them on other plates.

    python vcp_system.py --plate BR00117019 --wells B16 M20 C05 --dmso 4 --chips img1.png --out reports/demo
    python vcp_system.py --chips path/to/chip_photos/*.png --out reports/chips

Inputs : CPJUMP1 wells streamed from the public archive (3 bright-field planes per field; the 5 real stains are also
         downloaded but used ONLY for the "real-stain check" columns, never by the models); chip photos (PNG/JPG/TIFF).
Outputs: <out>/report.html (tables + images), <out>/report.json.
"""
import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
MODELS = ROOT / "models"
STAINS = ["DNA", "ER", "RNA", "AGP", "Mito"]
# SHA-256 of models/q2_final.pkl as written by tools/fit_final_models.py. A pickle can run code when loaded, so the file is only
# unpickled if it matches; otherwise the same winning paths are refitted from results/ instead (update this after a refit).
Q2_SHA256 = "d651bd94366851e631a30d56d497586f484cb976a903fa4c1b00986454e98086"


def load_q2():
    """The saved Q2 pipelines when the installed scikit-learn is the one that saved them; otherwise the same winning paths
    refitted on the calibration plate from results/ (deterministic, a few seconds), so the system runs with any version."""
    import hashlib
    import sklearn
    try:
        import cloudpickle
        raw = (MODELS / "q2_final.pkl").read_bytes()
        if hashlib.sha256(raw).hexdigest() != Q2_SHA256:
            raise ValueError("models/q2_final.pkl does not match its recorded SHA-256, so it is not unpickled")
        saved = cloudpickle.loads(raw)
        if saved["sklearn"] == sklearn.__version__:
            return saved["models"]
        print(f"Q2: models saved with scikit-learn {saved['sklearn']}, installed {sklearn.__version__}: refitting them", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"Q2: saved models not loadable here ({type(e).__name__}: {e}): refitting them", flush=True)
    from tools.fit_final_models import fit_q2
    return fit_q2()[0]


def _arrays(d):
    return {k: (np.array(v) if isinstance(v, list) and k not in ("cols", "states", "held_out_experiments") else v) for k, v in d.items()}


class System:
    def __init__(self, device=None, tta=None):
        import torch
        import data as D
        from weights import winner_json
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tta = (self.dev != "cpu") if tta is None else tta  # the validated mode; 8x slower, so default on GPU only
        w = json.loads(Path(winner_json()).read_text())
        ms = [D.load_model(Path(d), dev=self.dev) for d in w["run_dirs"]]
        self.models, self.cfg = [m for m, _ in ms], ms[0][1]
        self.norm = json.loads((ROOT / "runs" / "winner" / "norm.json").read_text())
        self.q2 = load_q2()
        self.q3 = _arrays(json.loads((MODELS / "q3_states.json").read_text()))
        self.q4 = _arrays(json.loads((MODELS / "q4_chip_qc.json").read_text()))
        self._emb = None

    # ---------------------------------------------------------------------------------------------- Q1
    def stain(self, x):
        """x: normalised field (8, H, W) or bright-field (3, H, W) -> virtual stains (5, H, W)."""
        import data as D
        return D.predict_multi(self.models, self.cfg, x, self.dev, self.tta)

    # ---------------------------------------------------------------------------------------------- Q2
    @staticmethod
    def field_features(virt):
        from evaluate import site_features
        return {f"virt_{k}": v for k, v in site_features(virt, STAINS, virt[0]).items()}

    def drug_response(self, wells, dmso_rows=None):
        """wells: DataFrame, one row per well (mean of its fields' virtual-stain features).
        dmso_rows: boolean mask of DMSO control wells of the same plate (>= 3: used for the plate normalisation that one
        pipeline needs; otherwise the calibration plate's DMSO statistics are used). Returns one row per well."""
        out = pd.DataFrame(index=wells.index)
        for label, m in self.q2.items():
            X = wells[m["names"]].to_numpy(float)
            plate = np.zeros(len(X), int)
            stats = getattr(m["fitted"].pre.clean_, "stats", None)
            if stats is not None and dmso_rows is not None and dmso_rows.sum() >= 3:
                Dm = X[dmso_rows]
                med = np.median(Dm, 0)
                mad = 1.4826 * np.median(np.abs(Dm - med), 0)
                stats[1] = (med, np.where(mad < 1e-12, Dm.std(0) + 1e-12, mad))
                plate[:] = 1
            s = m["fitted"].score(X, plate)
            if stats is not None:
                stats.pop(1, None)
            if m["task"] == "reg":
                out["viability_estimate"] = np.clip(s, 0, 1.5)
            else:
                out[f"{label}_score"] = s
                out[f"{label}_call"] = s >= m["threshold"]
        out["treated_vs_dmso_applies"] = ~out["toxic_0.8_call"]  # the drug-effect model was fitted on non-toxic wells only
        out = out.drop(columns=["treated_vs_dmso_call"])  # weak model (held-out AUC 0.73): a score, not a yes/no call
        return out

    # ---------------------------------------------------------------------------------------------- Q3
    def cells(self, stains):
        """Segment cells on (virtual or real) stains with the winning Q3 path; assign cell states."""
        from sweep.q3_search import assign, cell_features, segment
        q = self.q3
        mask = segment(stains, q["seg_path"])
        t = cell_features(mask, stains, with_haralick=False, with_zernike=False)
        if len(t) == 0:
            return mask, np.zeros(0, int)
        A = np.clip((np.nan_to_num(t[q["cols"]].to_numpy(float)) - q["med"]) / q["mad"], -20, 20)
        Z = (A - q["pca_mean"]) @ q["pca_components"].T
        return mask, assign(Z.astype(np.float32), q["centroids"])

    def state_names(self):
        """Short names given to the states from their mean size and shape (the states themselves come from the data)."""
        names = []
        for s in self.q3["states"]:
            size = "very large" if s["area"] > 2000 else "large" if s["area"] > 800 else "small"
            shape = "elongated" if s["eccentricity"] > 0.78 or s["solidity"] < 0.86 else "compact"
            names.append(f"S{s['state']} {size} {shape}")
        return names

    def state_legend(self):
        return "; ".join(f"{n}: mean area {s['area']:.0f} px (≈ {s['area'] * 1.69:.0f} µm²), eccentricity {s['eccentricity']:.2f}, "
                         f"solidity {s['solidity']:.2f}" for n, s in zip(self.state_names(), self.q3["states"]))

    # ---------------------------------------------------------------------------------------------- Q4
    def chip_quality(self, image):
        """image: path or PIL image of an organ-on-a-chip photo -> P(good chip)."""
        from PIL import Image
        from sweep import q4_parts as P
        im = Image.open(image).convert("L") if isinstance(image, (str, Path)) else image.convert("L")
        centre = P.variants(im)["centre_crop"][0]
        small = np.asarray(centre.resize(P.FEAT_HW[::-1], Image.BILINEAR)).astype(np.float32)
        hc = P.channel_features(small)
        if self._emb is None:
            self._emb = P.Embedder(self.q4["path"]["representation"], self.dev)
        hw = P.DEEP[self.q4["path"]["representation"]][2]
        deep = self._emb(np.asarray(centre.resize(hw[::-1], Image.BILINEAR), np.uint8)[None])[0]
        q = self.q4
        x = np.concatenate([deep, hc]).astype(float)
        x = np.where(np.isfinite(x), x, q["impute"])                      # imputer -> scaler -> correlation filter -> L1 logistic
        z = ((x - q["mean"]) / q["scale"])[q["keep"].astype(int)]
        return float(1 / (1 + np.exp(-(z @ q["coef"] + q["intercept"]))))


# -------------------------------------------------------------------------------------------------- CLI
def _png(rgb_or_grey, size=220):
    from PIL import Image
    a = np.asarray(rgb_or_grey)
    if a.dtype != np.uint8:
        a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(a)
    im.thumbnail((size, size))
    b = io.BytesIO()
    im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def composite(st, ref, mask=None):
    from skimage.segmentation import find_boundaries
    rgb = np.stack([np.clip((st[c] - np.percentile(ref[c], 1)) / (np.percentile(ref[c], 99.7) - np.percentile(ref[c], 1) + 1e-6), 0, 1)
                    for c in (3, 1, 0)], -1)
    if mask is not None:
        rgb[find_boundaries(mask, mode="outer")] = (1, 1, 1)
    return rgb


def run_plate(sysm, plate, wells, n_dmso, sites, log=print):
    from download import build_wells, fetch, load_site
    import config as C
    from evaluate import nuclei
    meta = build_wells().set_index("well")
    if n_dmso:
        wells = list(wells) + [w for w in meta[meta.pert_iname == "DMSO"].index if w not in wells][:n_dmso]
    ld = pd.read_csv(io.BytesIO(fetch(f"{C.S3}/{C.PILOT}/load_data_csv/{C.BATCH}/{plate}/load_data.csv")))
    n = sysm.norm.get(plate)
    rows, fields = [], []
    have = set(ld.Metadata_Site.unique())
    if not set(sites) & have:
        raise SystemExit(f"none of the fields {sites} exist on {plate} (it has fields {sorted(have)})")
    if set(sites) - have:
        log(f"  fields {sorted(set(sites) - have)} do not exist on {plate}; using {sorted(set(sites) & have)}")
        sites = [x for x in sites if x in have]
    if n is None:
        log(f"  {plate} has no stored normalisation: each field is scaled by its own percentiles (training used whole-plate "
            "percentiles, so expect some shift)")
    for w in wells:
        feats = []
        for site in sites:
            r = ld[(ld.Metadata_Well == w) & (ld.Metadata_Site == site)].iloc[0]
            raw = load_site(r).astype(np.float32)
            if n is None:  # a plate without stored normalisation: percentiles of this field
                lo_, hi_ = np.percentile(raw, 0.5, axis=(1, 2)), np.percentile(raw, 99.9, axis=(1, 2))
            else:
                lo_, hi_ = np.array(n["lo"], np.float32), np.array(n["hi"], np.float32)
            x = (raw - lo_[:, None, None]) / (hi_ - lo_)[:, None, None]
            virt = sysm.stain(x)                                   # Q1: from the 3 bright-field planes only
            feats.append(sysm.field_features(virt))
            mask, states = sysm.cells(virt)                        # Q3 on the virtual stains
            fields.append({"well": w, "site": site, "x": x, "virt": virt, "mask": mask, "states": states,
                           "real_nuclei": nuclei(x[0])[0]})        # real stain: check column only
            log(f"  {w} field {site}: {int(mask.max())} cells (virtual stains)")
        f = pd.DataFrame(feats).mean().to_dict()
        rows.append({"well": w, "compound": meta.loc[w, "pert_iname"], "split": meta.loc[w, "split"], **f})
    wells_df = pd.DataFrame(rows)
    dmso = (wells_df.compound == "DMSO").to_numpy()
    q2 = sysm.drug_response(wells_df, dmso)                         # Q2
    res = pd.concat([wells_df[["well", "compound", "split"]], q2], axis=1)
    # real-stain check (not used by any model): viability = real nuclei / median of the DMSO wells in this run
    real = pd.DataFrame([{"well": f["well"], "n": f["real_nuclei"]} for f in fields]).groupby("well").n.mean()
    if dmso.sum():
        res["viability_real_stains"] = res.well.map(real / real[res.well[dmso]].median()).round(2)
    names = sysm.state_names()
    for k, nm in enumerate(names):
        res[nm] = res.well.map(lambda w: np.mean(np.concatenate([f["states"] for f in fields if f["well"] == w]) == k)
                               if sum(len(f["states"]) for f in fields if f["well"] == w) else np.nan).round(2)
    res["cells_per_field"] = res.well.map(lambda w: np.mean([int(f["mask"].max()) for f in fields if f["well"] == w]))
    return res, fields


def html_report(out, res, fields, chips, sysm, args):
    rows = []
    for f in fields:
        bf = f["x"][6]
        bfn = np.clip((bf - np.percentile(bf, 1)) / (np.percentile(bf, 99.5) - np.percentile(bf, 1) + 1e-6), 0, 1)
        rows.append(f"<tr><td>{f['well']} / field {f['site']}</td><td><img src='{_png(bfn)}'></td>"
                    f"<td><img src='{_png(composite(f['virt'], f['x'][:5], f['mask']))}'></td>"
                    f"<td><img src='{_png(composite(f['x'][:5], f['x'][:5]))}'></td></tr>")
    chip_rows = "".join(f"<tr><td>{Path(c['image']).name}</td><td><img src='{c['thumb']}'></td><td>{c['p_good']:.3f}</td>"
                        f"<td>{'good' if c['p_good'] >= 0.5 else 'bad'}</td></tr>" for c in chips)
    style = ("<style>body{font-family:sans-serif;margin:24px;color:#111}table{border-collapse:collapse;margin:12px 0}"
             "td,th{border:1px solid #ddd;padding:4px 8px;font-size:13px}th{background:#f4f4f2}</style>")
    html = (f"<html><head><meta charset='utf-8'>{style}<title>VCP system report</title></head><body>"
            f"<h1>Virtual Cell Painting: end-to-end report</h1><p>Command: <code>{' '.join(sys.argv)}</code><br>"
            f"Q1 inference: 2-seed ensemble{' + test-time augmentation' if sysm.tta else ''} on {sysm.dev}. "
            "Q2/Q3 models fitted on calibration plate BR00117017. Real stains are shown only as a check; no model uses them. "
            "The drug-effect score (non-toxic wells only) is low confidence: held-out AUC 0.73.<br>"
            f"Cell states (names ours, from the measured means): {sysm.state_legend()}</p>"
            + (f"<h2>Q2 drug response and Q3 cell states, per well</h2>{res.round(3).to_html(index=False)}"
               "<h2>Fields: bright-field input, virtual stains with Q3 cell outlines, real stains (check)</h2>"
               f"<table><tr><th>well</th><th>bright-field</th><th>virtual + cells</th><th>real (check)</th></tr>{''.join(rows)}</table>"
               if res is not None else "")
            + (f"<h2>Q4 chip quality</h2><table><tr><th>image</th><th></th><th>P(good)</th><th>call</th></tr>{chip_rows}</table>"
               if chips else "") + "</body></html>")
    (out / "report.html").write_text(html)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--plate", help="CPJUMP1 plate barcode, e.g. BR00117019 (an A549 plate never used for fitting)")
    ap.add_argument("--wells", nargs="*", default=[], help="wells to analyse, e.g. B16 M20")
    ap.add_argument("--dmso", type=int, default=4, help="add this many DMSO control wells of the plate (plate normalisation, check)")
    ap.add_argument("--sites", nargs="+", type=int, default=[1, 11], help="fields per well (the Q2 models were fitted on 2: sites 1, 11)")
    ap.add_argument("--chips", nargs="*", default=[], help="organ-on-a-chip photos for Q4")
    ap.add_argument("--tta", action="store_true", help="test-time augmentation for Q1 (default: on with a GPU)")
    ap.add_argument("--out", default="reports/run")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sysm = System(tta=True if a.tta else None)
    print(f"models loaded (Q1 on {sysm.dev}, test-time augmentation {sysm.tta})", flush=True)
    res, fields, chips = None, [], []
    if a.plate:
        print(f"Q1-Q3: plate {a.plate}, wells {a.wells} + {a.dmso} DMSO wells, fields {a.sites}", flush=True)
        res, fields = run_plate(sysm, a.plate, a.wells, a.dmso, a.sites)
        names = sysm.state_names()
        short = pd.DataFrame({
            "well": res.well, "compound": res.compound.str[:20],
            "kills>50%": np.where(res["toxic_0.5_call"], "yes", "no"), "kills>20%": np.where(res["toxic_0.8_call"], "yes", "no"),
            "viability": res.viability_estimate.round(2),
            "(real)": res.get("viability_real_stains", pd.Series(np.nan, index=res.index)),
            "drug effect*": np.where(~res.treated_vs_dmso_applies, "(toxic)", res.treated_vs_dmso_score.round(2).astype(str)),
            "cells": res.cells_per_field.round(0).astype(int),
            "main state": res[names].idxmax(axis=1).fillna("-").str.split(" ", n=1).str[1]})
        print("\nQ2 drug response + Q3 cells per well (virtual stains only; (real) = real-stain check):")
        print(short.to_string(index=False))
        print("* drug-effect score for non-toxic wells: low confidence (held-out AUC 0.73), so no yes/no call")
    for c in a.chips:
        from PIL import Image
        p = sysm.chip_quality(c)
        im = Image.open(c).convert("L")
        chips.append({"image": str(c), "p_good": p, "thumb": _png(np.asarray(im.resize((im.width // 8, im.height // 8))))})
        print(f"Q4 chip {Path(c).name}: P(good) = {p:.3f} -> {'good' if p >= 0.5 else 'bad'}", flush=True)
    rep = {"plate": a.plate, "wells": None if res is None else res.round(4).to_dict("records"),
           "chips": [{k: v for k, v in c.items() if k != "thumb"} for c in chips],
           "q1": {"tta": sysm.tta, "device": sysm.dev}}
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    html_report(out, res, fields, chips, sysm, a)
    print("report:", out / "report.html")


if __name__ == "__main__":
    main()
