"""Q2 building blocks: per-node options of the search tree (cleaning, transform, selection, reduction, model,
calibration, imbalance) and the leakage-safe per-fold fit used by sweep/q2_toxicity.py.

Everything that learns is fitted on the training rows passed to `fit_fold` only:
  cleaning (plate robust-z uses DMSO wells of each plate among the training rows + label-free reference DMSO
  wells that are never samples), transform, selection, reduction, model hyperparameters (inner grouped 3-fold CV,
  <= 3 values per model), calibration (CalibratedClassifierCV on inner grouped splits), decision threshold.

Inputs per row: X (features), plate (0/1), is_dmso (bool), group (compound / DMSO well id).
Task 'clf' (binary, scored by AUC) or 'reg' (viability, scored by R^2; calibration / imbalance not applicable).
"""
import inspect
import warnings

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor,
                              StackingClassifier, StackingRegressor)
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedGroupKFold, GroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC, LinearSVR

warnings.filterwarnings("ignore")  # convergence chatter from small inner fits

SEED = 0
CLEANING = ["none", "near_zero_variance", "winsorise", "plate_robust_z_vs_dmso"]
TRANSFORM = ["zscore", "robust_z", "yeo_johnson"]
SELECTION = ["none", "correlation_filter_0.9", "kbest_mutual_info", "l1_lasso", "stability_selection",
             "drop_channel_block"]
REDUCTION = ["none", "pca_95", "pls"]
MODELS = ["logistic_unpenalised", "logistic_l1", "logistic_l2", "logistic_elasticnet", "mixed_logistic_plate",
          "lda", "naive_bayes", "svm_linear", "svm_rbf", "knn", "random_forest", "extra_trees",
          "hist_gradient_boosting", "xgboost", "mlp", "stacking_best_per_family"]
REG_EXCLUDED = {"lda", "naive_bayes"}  # no regression counterpart -> not in the viability tree
CALIBRATION = ["none", "platt", "isotonic"]
IMBALANCE = ["class_weights", "threshold_tuning"]
BASELINE = {"cleaning": "none", "transform": "zscore", "selection": "none", "reduction": "none",
            "model": "logistic_l2", "calibration": "none", "imbalance": "class_weights"}
DROP_BLOCK = "_Mito"  # deterministic channel block dropped by drop_channel_block (stain sources only)


def tree_for(source, task):
    t = {"cleaning": CLEANING, "transform": TRANSFORM,
         "selection": [s for s in SELECTION if not (s == "drop_channel_block" and source == "brightfield")],
         "reduction": REDUCTION, "model": MODELS if task == "clf" else [m for m in MODELS if m not in REG_EXCLUDED]}
    if task == "clf":
        t.update(calibration=CALIBRATION, imbalance=IMBALANCE)
    return t


def baseline_for(task):
    return {k: v for k, v in BASELINE.items() if task == "clf" or k in ("cleaning", "transform", "selection",
                                                                         "reduction", "model")}


# ------------------------------------------------------------------------------------------ sklearn version shims
_LR_NEW = inspect.signature(LogisticRegression).parameters["penalty"].default == "deprecated"
_MLP_SW = "sample_weight" in inspect.signature(MLPClassifier.fit).parameters


def logreg(kind="l2", C=1.0, class_weight=None):
    if _LR_NEW:
        kw = {"none": dict(C=np.inf), "l2": dict(C=C), "l1": dict(C=C, l1_ratio=1.0, solver="liblinear"),
              "en": dict(C=C, l1_ratio=0.5, solver="saga")}[kind]
    else:
        kw = {"none": dict(penalty=None), "l2": dict(C=C), "l1": dict(C=C, penalty="l1", solver="liblinear"),
              "en": dict(C=C, penalty="elasticnet", l1_ratio=0.5, solver="saga")}[kind]
    return LogisticRegression(max_iter=3000 if kind == "en" else 2000, class_weight=class_weight, **kw)


# ------------------------------------------------------------------------------------------ cleaning
def clean_fit(name, X, plate, is_dmso, ref=None):
    """Return (function X, plate -> cleaned X, kept column indices), learned on the given (training) rows only."""
    allc = np.arange(X.shape[1])
    if name == "none":
        return (lambda A, p: A), allc
    if name == "near_zero_variance":
        sd = X.std(0)
        top = np.array([np.max(np.unique(c, return_counts=True)[1]) / len(c) for c in X.T])
        keep = np.where((sd > 1e-8 * (np.abs(X.mean(0)) + 1e-12)) & (top < 0.95))[0]
        keep = keep if len(keep) else allc
        return (lambda A, p: A[:, keep]), keep
    if name == "winsorise":
        lo, hi = np.percentile(X, [1, 99], axis=0)
        return (lambda A, p: np.clip(A, lo, hi)), allc
    if name == "plate_robust_z_vs_dmso":
        stats_ = {}
        for pl in (0, 1):
            parts = [X[(plate == pl) & is_dmso]]
            if ref is not None and len(ref[0]):
                parts.append(ref[0][ref[1] == pl])
            D = np.vstack(parts)
            if len(D) < 3:  # no DMSO of this plate in the training rows -> fall back to all training rows
                D = X[plate == pl] if (plate == pl).sum() >= 3 else X
            med = np.median(D, 0)
            mad = 1.4826 * np.median(np.abs(D - med), 0)
            mad = np.where(mad < 1e-12, D.std(0) + 1e-12, mad)
            stats_[pl] = (med, mad)

        def f(A, p):
            out = np.empty_like(A, dtype=float)
            for pl, (med, mad) in stats_.items():
                m = p == pl
                out[m] = (A[m] - med) / mad
            return out
        f.stats = stats_  # per-plate DMSO statistics: a deployed model can add a new plate's (vcp_system.py)
        return f, allc
    raise ValueError(name)


def make_transform(name):
    return {"zscore": StandardScaler(), "robust_z": RobustScaler(),
            "yeo_johnson": PowerTransformer(method="yeo-johnson", standardize=True)}[name]


# ------------------------------------------------------------------------------------------ selection
def _top(scores, k):
    return np.argsort(-np.asarray(scores))[:k]


def _l1_coefs(X, y, task, rng=None):
    if task == "clf":
        m = logreg("l1", C=0.1, class_weight="balanced").fit(X, y)
        return np.abs(m.coef_.ravel())
    yc = y - y.mean()
    amax = np.max(np.abs(X.T @ yc)) / len(y) + 1e-12
    return np.abs(Lasso(alpha=0.1 * amax, max_iter=5000).fit(X, y).coef_)


def select_fit(name, X, y, task, names):
    """Return the column indices kept (learned on training rows). X is already transformed (standardised)."""
    p = X.shape[1]
    allc = np.arange(p)
    if name == "none":
        return allc
    if name == "drop_channel_block":
        keep = np.array([i for i, n in enumerate(names) if DROP_BLOCK not in n])
        return keep if len(keep) else allc
    if name == "correlation_filter_0.9":
        C = np.abs(np.corrcoef(X, rowvar=False))
        C = np.nan_to_num(C, nan=0.0)
        keep = []
        for j in range(p):  # greedy, column order: keep j unless |r| > 0.9 with an already kept column
            if not keep or C[j, keep].max() <= 0.9:
                keep.append(j)
        return np.array(keep)
    if name == "kbest_mutual_info":
        k = min(30, p)
        fn = mutual_info_classif if task == "clf" else mutual_info_regression
        return np.sort(_top(fn(X, y, random_state=SEED), k))
    if name == "l1_lasso":
        c = _l1_coefs(X, y, task)
        keep = np.where(c > 1e-8)[0]
        return keep if len(keep) >= 1 else np.sort(_top(c, 5))
    if name == "stability_selection":
        rng = np.random.default_rng(SEED)
        freq = np.zeros(p)
        n_res = 20
        for _ in range(n_res):
            ii = rng.integers(0, len(y), len(y))
            if task == "clf" and len(np.unique(y[ii])) < 2:
                continue
            freq += _l1_coefs(X[ii], y[ii], task) > 1e-8
        freq /= n_res
        keep = np.where(freq >= 0.6)[0]
        return keep if len(keep) >= 1 else np.sort(_top(freq, 5))
    raise ValueError(name)


class PLSTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=10):
        self.n_components = n_components

    def fit(self, X, y):
        k = max(1, min(self.n_components, X.shape[1], X.shape[0] - 1))
        self.pls_ = PLSRegression(n_components=k, scale=False).fit(X, np.asarray(y, float))
        return self

    def transform(self, X):
        return self.pls_.transform(X)


def make_reduction(name):
    return {"none": None, "pca_95": PCA(n_components=0.95, svd_solver="full", random_state=SEED),
            "pls": PLSTransformer(10)}[name]


class Preprocessor:
    """cleaning -> transform -> selection -> reduction, all learned in fit() on the training rows."""

    def __init__(self, path, task, names, ref=None):
        self.path, self.task, self.names, self.ref = path, task, list(names), ref

    def fit(self, X, y, plate, is_dmso):
        self.clean_, self.cols_ = clean_fit(self.path["cleaning"], X, plate, is_dmso, self.ref)
        A = self.clean_(X, plate)
        self.tf_ = clone(make_transform(self.path["transform"])).fit(A)
        A = np.nan_to_num(self.tf_.transform(A))
        self.sel_ = select_fit(self.path["selection"], A, y, self.task, [self.names[i] for i in self.cols_])
        A = A[:, self.sel_]
        red = make_reduction(self.path["reduction"])
        if red is not None and self.path["reduction"] == "pca_95" and A.shape[1] < 2:
            red = None
        self.red_ = clone(red).fit(A, y) if red is not None else None
        return self

    def transform(self, X, plate):
        A = np.nan_to_num(self.tf_.transform(self.clean_(X, plate)))[:, self.sel_]
        return self.red_.transform(A) if self.red_ is not None else A

    def selected_features(self):
        return [self.names[self.cols_[i]] for i in self.sel_]


# ------------------------------------------------------------------------------------------ models
def make_model(name, task, cw, smoke=False, n_pos=None, n_neg=None):
    """Returns (estimator, grid dict param -> values). cw=True -> class weighting where the model supports it."""
    bal = "balanced" if (cw and task == "clf") else None
    Cs = [0.3] if smoke else [0.03, 0.3, 3.0]
    if task == "clf":
        spw = (n_neg / max(n_pos, 1)) if (cw and n_pos) else 1.0
        prior = [0.5, 0.5] if cw else None
        lr_l2 = logreg("l2", 0.3, bal)
        hgb = HistGradientBoostingClassifier(learning_rate=0.1, max_leaf_nodes=15, max_iter=150,
                                             class_weight=bal, random_state=SEED)
        mlp = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=300, early_stopping=True,
                            random_state=SEED)
        table = {
            "logistic_unpenalised": (logreg("none", 1.0, bal), {}),
            "logistic_l1": (logreg("l1", 0.3, bal), {"C": Cs}),
            "logistic_l2": (lr_l2, {"C": Cs}),
            "logistic_elasticnet": (logreg("en", 0.3, bal), {"C": Cs}),
            # mixed_logistic_plate: with 2 plates a plate random-effect variance is not identifiable -> GLM with a
            # plate fixed effect (indicator column appended after preprocessing, see fit_fold), L2 C tuned
            "mixed_logistic_plate": (logreg("l2", 0.3, bal), {"C": Cs}),
            "lda": (LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto", priors=prior), {}),
            "naive_bayes": (GaussianNB(priors=prior), {}),
            "svm_linear": (LinearSVC(C=0.1, class_weight=bal, max_iter=5000), {"C": [0.1] if smoke else [0.01, 0.1, 1.0]}),
            "svm_rbf": (SVC(C=1.0, gamma="scale", class_weight=bal), {"C": [1.0] if smoke else [0.3, 1.0, 3.0]}),
            "knn": (KNeighborsClassifier(n_neighbors=15), {"n_neighbors": [15] if smoke else [5, 15, 45]}),
            "random_forest": (RandomForestClassifier(n_estimators=50 if smoke else 200, class_weight=bal,
                                                     random_state=SEED, n_jobs=1),
                              {"min_samples_leaf": [5] if smoke else [1, 5, 20]}),
            "extra_trees": (ExtraTreesClassifier(n_estimators=50 if smoke else 200, class_weight=bal,
                                                 random_state=SEED, n_jobs=1),
                            {"min_samples_leaf": [5] if smoke else [1, 5, 20]}),
            "hist_gradient_boosting": (hgb, {"learning_rate": [0.1] if smoke else [0.03, 0.1, 0.3]}),
            "xgboost": (_xgb("clf", spw, smoke), {"max_depth": [3] if smoke else [2, 4, 6]}),
            "mlp": (mlp, {"hidden_layer_sizes": [(64,)] if smoke else [(64,), (128, 64), (128, 64, 32)]}),
            "stacking_best_per_family": (StackingClassifier(
                [("lr", lr_l2), ("hgb", hgb), ("mlp", mlp)], final_estimator=logreg("l2", 1.0),
                cv=StratifiedKFold(3, shuffle=True, random_state=SEED), n_jobs=1), {}),
        }
    else:
        hgb = HistGradientBoostingRegressor(learning_rate=0.1, max_leaf_nodes=15, max_iter=150, random_state=SEED)
        mlp = MLPRegressor(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=300, early_stopping=True, random_state=SEED)
        As = [1.0] if smoke else [0.1, 3.0, 100.0]
        La = [0.003] if smoke else [0.0003, 0.003, 0.03]
        table = {
            "logistic_unpenalised": (LinearRegression(), {}),
            "logistic_l1": (Lasso(alpha=0.003, max_iter=5000), {"alpha": La}),
            "logistic_l2": (Ridge(alpha=1.0), {"alpha": As}),
            "logistic_elasticnet": (ElasticNet(alpha=0.003, l1_ratio=0.5, max_iter=5000), {"alpha": La}),
            "mixed_logistic_plate": (Ridge(alpha=1.0), {"alpha": As}),
            "svm_linear": (LinearSVR(C=0.1, max_iter=5000), {"C": [0.1] if smoke else [0.01, 0.1, 1.0]}),
            "svm_rbf": (SVR(C=1.0, gamma="scale"), {"C": [1.0] if smoke else [0.3, 1.0, 3.0]}),
            "knn": (KNeighborsRegressor(n_neighbors=15), {"n_neighbors": [15] if smoke else [5, 15, 45]}),
            "random_forest": (RandomForestRegressor(n_estimators=50 if smoke else 200, random_state=SEED, n_jobs=1),
                              {"min_samples_leaf": [5] if smoke else [1, 5, 20]}),
            "extra_trees": (ExtraTreesRegressor(n_estimators=50 if smoke else 200, random_state=SEED, n_jobs=1),
                            {"min_samples_leaf": [5] if smoke else [1, 5, 20]}),
            "hist_gradient_boosting": (hgb, {"learning_rate": [0.1] if smoke else [0.03, 0.1, 0.3]}),
            "xgboost": (_xgb("reg", 1.0, smoke), {"max_depth": [3] if smoke else [2, 4, 6]}),
            "mlp": (mlp, {"hidden_layer_sizes": [(64,)] if smoke else [(64,), (128, 64), (128, 64, 32)]}),
            "stacking_best_per_family": (StackingRegressor(
                [("lr", Ridge(alpha=3.0)), ("hgb", hgb), ("mlp", mlp)], final_estimator=Ridge(alpha=1.0),
                cv=KFold(3, shuffle=True, random_state=SEED), n_jobs=1), {}),
        }
    est, grid = table[name]
    return clone(est), grid


def _xgb(task, spw, smoke):
    import xgboost as xgb
    kw = dict(n_estimators=50 if smoke else 200, learning_rate=0.1, max_depth=3, subsample=0.8,
              colsample_bytree=0.8, tree_method="hist", n_jobs=1, random_state=SEED, verbosity=0)
    return xgb.XGBClassifier(scale_pos_weight=spw, eval_metric="logloss", **kw) if task == "clf" \
        else xgb.XGBRegressor(**kw)


def _fit(est, X, y, sample_weight=None):
    if sample_weight is not None:
        return est.fit(X, y, sample_weight=sample_weight)
    return est.fit(X, y)


def score_of(est, X, task):
    """Continuous score for ranking: P(y=1) if available, else the decision function; regression -> prediction."""
    if task == "reg":
        return est.predict(X)
    if hasattr(est, "predict_proba"):
        try:
            return est.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return est.decision_function(X)


def metric(y, s, task):
    if task == "reg":
        return float(r2_score(y, s))
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def grouped_splits(y, groups, k, task, seed):
    """Grouped (and for classification stratified) k-fold; k shrinks if there are too few groups / positives."""
    uniq = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(len(uniq))
    gp = np.searchsorted(uniq, groups)
    gp = perm[gp]  # random relabelling of groups = random fold assignment
    if task == "clf":
        npos_groups = len(np.unique(gp[y == 1]))
        nneg_groups = len(np.unique(gp[y == 0]))
        k = max(2, min(k, npos_groups, nneg_groups))
        return list(StratifiedGroupKFold(n_splits=k, shuffle=False).split(np.zeros(len(y)), y, gp))
    k = max(2, min(k, len(uniq)))
    return list(GroupKFold(n_splits=k).split(np.zeros(len(y)), y, gp))


def _youden_quantile(y, s):
    """Threshold maximising Youden J on (inner OOF) scores, expressed as the fraction of rows below it
    (rank-based, so it transfers to a calibrated / refitted model)."""
    fpr, tpr, thr = roc_curve(y, s)
    t = thr[int(np.argmax(tpr - fpr))]
    return float(np.mean(s < t))


def fit_fold(path, task, X, y, plate, is_dmso, groups, names, ref=None, smoke=False, seed=0,
             fixed_params=None, n_inner=3):
    """Fit one full path on the training rows. Returns a FittedPath with .score(X, plate) and diagnostics."""
    pre = Preprocessor(path, task, names, ref).fit(X, y, plate, is_dmso)
    Z = pre.transform(X, plate)
    mixed = path["model"] == "mixed_logistic_plate"
    if mixed:
        Z = np.column_stack([Z, plate])
    cw = task == "clf" and path.get("imbalance", "class_weights") == "class_weights"
    n_pos = int((y == 1).sum()) if task == "clf" else None
    est, grid = make_model(path["model"], task, cw, smoke, n_pos, len(y) - (n_pos or 0))
    sw = None
    if cw and path["model"] == "mlp" and _MLP_SW:  # MLP has no class_weight -> balanced sample weights
        sw = np.where(y == 1, 0.5 / max(y.mean(), 1e-6), 0.5 / max(1 - y.mean(), 1e-6))
    inner = grouped_splits(y, groups, n_inner, task, seed)
    best_params, inner_scores, q = {}, {}, None
    key = next(iter(grid), None)
    if fixed_params is not None:
        best_params = dict(fixed_params)
    elif key is not None and len(grid[key]) > 1:
        for v in grid[key]:
            oof = np.full(len(y), np.nan)
            for tr, te in inner:
                m = _fit(clone(est).set_params(**{key: v}), Z[tr], y[tr], None if sw is None else sw[tr])
                oof[te] = score_of(m, Z[te], task)
            inner_scores[str(v)] = metric(y, oof, task) if task == "reg" else _pooled_auc(y, oof, inner)
        best_v = max(grid[key], key=lambda v: np.nan_to_num(inner_scores[str(v)], nan=-1e9))
        best_params = {key: best_v}
    est = clone(est).set_params(**best_params)
    if task == "clf" and path.get("imbalance") == "threshold_tuning":
        oof = np.full(len(y), np.nan)
        for tr, te in inner:
            oof[te] = score_of(_fit(clone(est), Z[tr], y[tr]), Z[te], task)
        q = _youden_quantile(y, oof)
    cal = path.get("calibration", "none") if task == "clf" else "none"
    if cal != "none":
        model = CalibratedClassifierCV(est, method="sigmoid" if cal == "platt" else "isotonic", cv=inner)
        model.fit(Z, y) if sw is None else model.fit(Z, y, sample_weight=sw)
    else:
        model = _fit(est, Z, y, sw)
    thr = 0.5
    if task == "clf":
        s_tr = score_of(model, Z, task)
        thr = float(np.quantile(s_tr, q)) if q is not None else (0.5 if hasattr(model, "predict_proba") else 0.0)
    return FittedPath(pre, model, mixed, task, best_params, thr, pre.selected_features())


def _pooled_auc(y, s, splits):
    """Mean per-fold AUC (pooled AUC across folds mixes score scales of different fits)."""
    vals = [roc_auc_score(y[te], s[te]) for _, te in splits if len(np.unique(y[te])) == 2]
    return float(np.mean(vals)) if vals else float("nan")


class FittedPath:
    def __init__(self, pre, model, mixed, task, params, threshold, selected):
        self.pre, self.model, self.mixed, self.task = pre, model, mixed, task
        self.params, self.threshold, self.selected = params, threshold, selected

    def score(self, X, plate):
        Z = self.pre.transform(X, plate)
        if self.mixed:
            Z = np.column_stack([Z, plate])
        return score_of(self.model, Z, self.task)

    def has_proba(self):
        return hasattr(self.model, "predict_proba")
