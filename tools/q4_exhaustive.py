"""Q4 check: run EVERY valid path of the searched Q4 tree at 100% of the data and see where the guided search's winner ranks.

    VCP_Q4_EMB=<embeddings folder> python tools/q4_exhaustive.py --workers 10 [--max-minutes 75]
    -> results/chips/q4/q4_exhaustive.csv (one row per path, grouped five-fold out-of-fold AUC) and q4_exhaustive_summary.json

The searched tree is the set of options the guided search actually ran (results/chips/q4/q4_search_summary.json, option_coverage);
each path is scored exactly as in the search (sweep/q4_chip_qc.py, score_search at 100%), so the two are directly comparable.
Paths run in a random order (seed 0); with --max-minutes the run stops at that time limit, and the paths done are then an unbiased
random sample of the tree (the summary says how many). Results are saved as they come in.
"""
import argparse
import itertools
import json
import os
import sys
import random
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES = ROOT / "results" / "chips" / "q4"


def score(path):
    from sweep import q4_chip_qc as Q
    t0 = time.time()
    try:
        auc = Q.score_search(path, 1.0)
    except Exception as e:  # a path that cannot be fitted counts as failed, as in the search
        return {**path, "auc": float("nan"), "seconds": time.time() - t0, "error": str(e)[:200]}
    return {**path, "auc": float(auc), "seconds": time.time() - t0, "error": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default=os.environ.get("VCP_Q4_EMB"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--out", default="/tmp/q4_exhaustive")
    a = ap.parse_args()
    os.environ.update(Q4_EMB_DIR=str(a.emb_dir), Q4_OUT=str(a.out), Q4_SMOKE="0", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      MKL_NUM_THREADS="1")
    from sweep import q4_chip_qc as Q
    summ = json.loads((RES / "q4_search_summary.json").read_text())
    cov = summ["option_coverage"]
    nodes = {k: [o for o, c in v.items() if c > 0] for k, v in cov.items()}   # options the guided search actually ran
    rules = Q.Rules(Q._data().reps())
    paths = [dict(zip(nodes, combo)) for combo in itertools.product(*nodes.values())]
    valid = [p for p in paths if rules(p)]
    print(f"searched tree: {len(paths)} combinations, {len(valid)} valid under the rules", flush=True)
    random.Random(0).shuffle(valid)
    best_search = summ["best"][0] if isinstance(summ["best"], list) else summ["best"]
    bp = best_search.get("path", best_search)
    win = next(p for p in valid if all(p[k] == bp[k] for k in nodes))
    valid.remove(win)
    valid.insert(0, win)  # the guided search's winner first, so its rank is known even if the time limit stops the run
    t0, rows, todo, running = time.time(), [], list(valid), set()
    stop = t0 + a.max_minutes * 60 if a.max_minutes else float("inf")
    with ProcessPoolExecutor(a.workers) as ex:
        while (todo and time.time() < stop) or running:
            while todo and len(running) < 2 * a.workers and time.time() < stop:
                running.add(ex.submit(score, todo.pop(0)))
            done, running = wait(running, timeout=30, return_when=FIRST_COMPLETED)
            running = set(running)
            rows += [f.result() for f in done]
            if done and len(rows) % 50 < len(done):
                print(f"  {len(rows)}/{len(valid)} paths, {time.time() - t0:.0f} s", flush=True)
                pd.DataFrame(rows).to_csv(RES / "q4_exhaustive.csv", index=False)
            if time.time() >= stop:
                todo = []  # time limit: finish the paths already running, start no more
    d = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    d.insert(0, "rank", range(1, len(d) + 1))
    d.to_csv(RES / "q4_exhaustive.csv", index=False)
    keys = list(nodes)
    match = d[(d[keys] == pd.Series({k: bp[k] for k in keys})).all(axis=1)] if isinstance(bp, dict) else d.iloc[0:0]
    out = {"combinations": len(paths), "valid": len(valid), "run": len(d), "scored": int(d.auc.notna().sum()),
           "complete": len(d) == len(valid), "order": "random (seed 0)", "workers": a.workers, "wall_s": round(time.time() - t0, 1),
           "median_seconds_per_fitted_path": round(float(d.seconds[d.seconds > 0.05].median()), 2),  # cached paths take ~0 s
           "best_exhaustive": {**{k: d.loc[0, k] for k in keys}, "auc": round(float(d.loc[0, "auc"]), 4)},
           "search_winner": {**({k: bp[k] for k in keys} if isinstance(bp, dict) else {}),
                             "auc": round(float(match.auc.iloc[0]), 4) if len(match) else None,
                             "rank": int(match["rank"].iloc[0]) if len(match) else None},
           "paths_within_0.005_of_best": int((d.auc >= d.auc.max() - 0.005).sum())}
    (RES / "q4_exhaustive_summary.json").write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
