"""Guided path search shared by all research questions.

A path = one option per node of a tree (dict node -> list of options), filtered by rules.
The search, for a fixed budget of paths:
  1. warm-up  : balanced design so every option of every node is used >= `warmup_per_option` times
  2. guided   : Optuna TPE proposes the next paths from the results so far
  3. stages   : each path is scored at 25% -> 50% -> 100% of the data (nested subsets);
                the worse half at a checkpoint is stopped (successive halving)
  4. exclusion: an option is removed from its node after `strikes` bad results ("bad" = stopped at the
                first checkpoint) within its first `strike_window` uses. The baseline option is never removed
                and every node keeps >= 2 options. Removed options are logged.
  5. stop     : when `budget` paths have been evaluated (paths proposed with a removed option are skipped
                at no cost and do not count)

Usage (see sweep/q2_toxicity.py etc.):
    from sweep.engine import Search
    s = Search("q2_real_toxic", tree, rules=valid_fn, baseline={...}, budget=250, out_dir=...)
    s.run(score_fn, workers=4)        # score_fn(path: dict, fraction: float, report) -> float (higher = better)
"""
import itertools
import json
import multiprocessing as mp
import os
import random
import time
from pathlib import Path

import optuna

FRACTIONS = (0.25, 0.5, 1.0)


class Excluded(Exception):
    pass


def always_valid(path):
    return True


class Search:
    def __init__(self, name, tree, out_dir, budget, rules=None, baseline=None, warmup_per_option=3,
                 strikes=4, strike_window=6, fractions=FRACTIONS, seed=0):
        self.name, self.tree, self.budget = name, {k: list(v) for k, v in tree.items()}, budget
        self.rules = rules or always_valid  # must be module-level (picklable) for multi-process runs
        self.baseline = baseline or {k: v[0] for k, v in self.tree.items()}
        self.warmup_per_option, self.strikes, self.strike_window = warmup_per_option, strikes, strike_window
        self.fractions, self.seed = fractions, seed
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.storage_url = f"sqlite:///{self.out / f'{name}.db'}"

    # ------------------------------------------------------------------ study and warm-up
    def _storage(self):
        return optuna.storages.RDBStorage(self.storage_url, engine_kwargs={"connect_args": {"timeout": 120}})

    def _study(self, worker=0):
        sampler = optuna.samplers.TPESampler(seed=self.seed + 1000 * worker, multivariate=True,
                                             constant_liar=True, n_startup_trials=0)
        pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1, reduction_factor=2, min_early_stopping_rate=0)
        return optuna.create_study(study_name=self.name, storage=self._storage(), direction="maximize",
                                   sampler=sampler, pruner=pruner, load_if_exists=True)

    def warmup_paths(self):
        """Balanced design: each option of each node appears >= warmup_per_option times (rules respected)."""
        rng = random.Random(self.seed)
        n = self.warmup_per_option * max(len(v) for v in self.tree.values())
        cols = {}
        for k, opts in self.tree.items():
            col = (opts * (n // len(opts) + 1))[:n]
            rng.shuffle(col)
            cols[k] = col
        paths = [self.baseline] + [{k: cols[k][i] for k in self.tree} for i in range(n)]
        out = []
        for p in paths:  # repair rule violations by re-drawing the offending nodes
            for _ in range(200):
                if self.rules(p):
                    break
                k = rng.choice(list(self.tree))
                p = {**p, k: rng.choice(self.tree[k])}
            if self.rules(p):
                out.append(p)
        return out

    def init(self):
        study = self._study()
        if not study.trials:
            for p in self.warmup_paths():
                study.enqueue_trial(p, skip_if_exists=True)
        return study

    # ------------------------------------------------------------------ exclusion rule
    def excluded(self, study):
        uses = {(k, o): [] for k, v in self.tree.items() for o in v}
        for t in study.trials:
            if t.user_attrs.get("skipped") or t.state not in (optuna.trial.TrialState.COMPLETE,
                                                              optuna.trial.TrialState.PRUNED):
                continue
            bad = t.state == optuna.trial.TrialState.PRUNED and t.last_step == 1
            for k, o in t.params.items():
                if (k, o) in uses:
                    uses[(k, o)].append((t.number, bad))
        out = []
        for k, opts in self.tree.items():
            alive = list(opts)
            for o in opts:
                hist = sorted(uses[(k, o)])[: self.strike_window]
                n_bad = sum(b for _, b in hist)
                if o != self.baseline.get(k) and n_bad >= self.strikes and len(alive) > 2:
                    alive.remove(o)
                    out.append({"node": k, "option": o, "bad": n_bad, "uses": len(uses[(k, o)])})
        return out

    def n_evaluated(self, study):
        return sum(1 for t in study.trials if not t.user_attrs.get("skipped")
                   and t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED))

    # ------------------------------------------------------------------ running
    def _objective(self, score_fn):
        self._announced = set()
        def objective(trial):
            study = trial.study
            path = {k: trial.suggest_categorical(k, v) for k, v in self.tree.items()}
            if not self.rules(path):
                trial.set_user_attr("skipped", "invalid")
                raise optuna.TrialPruned()
            done = {tuple(sorted(t.params.items())) for t in study.trials if t.number != trial.number
                    and not t.user_attrs.get("skipped") and t.state in (optuna.trial.TrialState.COMPLETE,
                                                                         optuna.trial.TrialState.PRUNED,
                                                                         optuna.trial.TrialState.RUNNING)}
            if tuple(sorted(path.items())) in done:  # deterministic scoring: a repeat adds nothing
                trial.set_user_attr("skipped", "duplicate")
                raise optuna.TrialPruned()
            ex = self.excluded(study)
            dead = {(e["node"], e["option"]) for e in ex}
            if os.environ.get("VCP_VERBOSE_PATHS") == "1" and os.environ.get("SEARCH_WORKER", "0") == "0":
                for e in ex:  # announce each newly excluded option once (worker 0 only)
                    if (e["node"], e["option"]) not in self._announced:
                        self._announced.add((e["node"], e["option"]))
                        print(f"  ** option excluded: {e['node']} = {e['option']} ({e['bad']} bad results in its first uses)", flush=True)
            if any((k, o) in dead for k, o in path.items()):
                trial.set_user_attr("skipped", "excluded_option")
                raise optuna.TrialPruned()
            t0 = time.time()
            score, seen = None, []
            for step, frac in enumerate(self.fractions, 1):
                score = score_fn(path, frac, trial)
                trial.report(score, step)
                trial.set_user_attr(f"score_{int(frac * 100)}", score)
                trial.set_user_attr(f"seconds_{int(frac * 100)}", time.time() - t0)
                seen.append(f"{int(frac * 100)}% {score:.3f}")
                if step < len(self.fractions) and trial.should_prune():
                    self._say(trial, path, seen, "stopped (worse half)")
                    raise optuna.TrialPruned()
            self._say(trial, path, seen, "completed")
            return score
        return objective

    @staticmethod
    def _say(trial, path, seen, outcome):
        """One line per evaluated path when VCP_VERBOSE_PATHS=1 (used for the video demo)."""
        if os.environ.get("VCP_VERBOSE_PATHS") == "1":
            print(f"path {trial.number:>3}: {' > '.join(str(v) for v in path.values())}\n"
                  f"          {' -> '.join(seen)}  {outcome}", flush=True)

    def _worker(self, score_fn, worker, hours):
        os.environ["SEARCH_WORKER"] = str(worker)  # lets score_fn pick a device per worker (e.g. cuda:worker)
        if os.environ.get("VCP_VERBOSE_PATHS") == "1":
            optuna.logging.set_verbosity(optuna.logging.WARNING)  # one line per path instead (see _say)
        study = self._study(worker)
        deadline = time.time() + hours * 3600

        def done(study_):
            recent = study_.trials[-100:]
            exhausted = len(recent) == 100 and all(t.user_attrs.get("skipped") for t in recent)
            return self.n_evaluated(study_) >= self.budget or time.time() > deadline or exhausted

        def stop(study_, _trial):
            if done(study_):
                study_.stop()

        while True:
            try:
                study.optimize(self._objective(score_fn), callbacks=[stop], catch=(Exception,),
                               timeout=max(1.0, deadline - time.time()))
                return
            except ValueError as e:  # parallel workers: a trial already closed by optuna ("Cannot tell a PRUNED trial")
                if "Cannot tell" not in str(e) or done(study):
                    if "Cannot tell" in str(e):
                        return
                    raise

    def run(self, score_fn, workers=1, hours=10.0):
        """score_fn must be importable (module-level) when workers > 1."""
        self.init()
        if workers == 1:
            self._worker(score_fn, 0, hours)
        else:
            for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(v, "1")  # one core per worker process (inherited by the spawned workers)
            ctx = mp.get_context("spawn")
            procs = [ctx.Process(target=self._worker, args=(score_fn, w, hours)) for w in range(workers)]
            [p.start() for p in procs]
            [p.join() for p in procs]
        return self.report()

    # ------------------------------------------------------------------ outputs
    def report(self):
        study = self._study()
        rows = []
        for t in study.trials:
            rows.append({"trial": t.number, "state": t.state.name, "skipped": t.user_attrs.get("skipped", ""),
                         "value": t.value, "stopped_at": t.last_step, **t.params,
                         **{k: v for k, v in t.user_attrs.items() if k.startswith(("score_", "seconds_"))}})
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_csv(self.out / f"{self.name}_paths.csv", index=False)
        ex = self.excluded(study)
        ev = df[(df.skipped == "") & df.state.isin(["COMPLETE", "PRUNED"])] if len(df) else df
        best = ev.sort_values("value", ascending=False).head(10) if len(ev) else ev
        coverage = {k: {str(o): int((ev[k] == o).sum()) for o in v} for k, v in self.tree.items()} if len(ev) else {}
        summary = {"name": self.name, "budget": self.budget, "evaluated": int(len(ev)),
                   "completed_100pct": int((ev.state == "COMPLETE").sum()) if len(ev) else 0,
                   "skipped_excluded": int((df.skipped == "excluded_option").sum()) if len(df) else 0,
                   "skipped_duplicate": int((df.skipped == "duplicate").sum()) if len(df) else 0,
                   "excluded_options": ex, "option_coverage": coverage,
                   "best": best.drop(columns=["skipped"]).to_dict("records") if len(best) else []}
        (self.out / f"{self.name}_summary.json").write_text(json.dumps(summary, indent=1, default=str))
        return summary


def nested_subset(n_or_groups, fraction, seed=0):
    """Indices of a nested subset: the 25% subset is inside the 50%, inside 100%.
    Pass an array of group labels to sample whole groups (e.g. compounds, experiments)."""
    import numpy as np
    labels = np.asarray(n_or_groups) if not isinstance(n_or_groups, int) else np.arange(n_or_groups)
    uniq = np.unique(labels)
    order = np.random.default_rng(seed).permutation(len(uniq))
    keep = set(uniq[order[: max(1, int(round(fraction * len(uniq))))]])
    return np.where(np.isin(labels, list(keep)))[0]


if __name__ == "__main__":  # self-test on a toy problem with a known best path and a known bad option
    import tempfile
    tree = {"a": ["x", "y", "z", "bad"], "b": [1, 2, 3, 4, 5, 6]}

    def score(path, frac, trial):
        s = {"x": 0.5, "y": 0.7, "z": 0.6, "bad": 0.1}[path["a"]] + 0.05 * path["b"]
        return s + random.gauss(0, 0.01) * (1 - frac)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    with tempfile.TemporaryDirectory() as d:
        s = Search("toy", tree, d, budget=40, baseline={"a": "x", "b": 1})
        w = s.warmup_paths()
        cov = {o: sum(p["a"] == o for p in w) for o in tree["a"]}
        assert all(c >= 3 for c in cov.values()), cov
        r = s.run(score)
        print("evaluated", r["evaluated"], "excluded", r["excluded_options"], "best", r["best"][0]["a"], r["best"][0]["b"])
        assert r["evaluated"] <= 24 and r["best"][0]["a"] == "y"  # 4 x 6 distinct paths: no duplicates evaluated
        assert any(e["option"] == "bad" for e in r["excluded_options"]), "bad option should be excluded"
        assert all(e["option"] != "x" for e in r["excluded_options"]), "baseline must never be excluded"
    print("engine self-test OK")
