"""Agreement and classification statistics shared by all research questions."""
import numpy as np
from scipy import stats as st
from sklearn.metrics import brier_score_loss, roc_auc_score


def bootstrap_ci(fn, *arrays, groups=None, n=2000, seed=0, alpha=0.05):
    """Point estimate and percentile CI of fn(*arrays). Resamples rows, or whole groups if given."""
    arrays = [np.asarray(a) for a in arrays]
    rng = np.random.default_rng(seed)
    if groups is None:
        idx_sets = [np.arange(len(arrays[0]))]
        draw = lambda: rng.integers(0, len(arrays[0]), len(arrays[0]))
    else:
        groups = np.asarray(groups)
        members = [np.where(groups == g)[0] for g in np.unique(groups)]
        draw = lambda: np.concatenate([members[i] for i in rng.integers(0, len(members), len(members))])
    vals = []
    for _ in range(n):
        ii = draw()
        try:
            vals.append(fn(*[a[ii] for a in arrays]))
        except ValueError:  # e.g. a resample with one class only
            continue
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"estimate": float(fn(*arrays)), "ci_low": float(lo), "ci_high": float(hi)}


def lins_ccc(x, y):
    """Lin's concordance correlation: agreement with the identity line (penalises bias and scale)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    sxy = np.mean((x - x.mean()) * (y - y.mean()))
    return float(2 * sxy / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))


def icc_2_1(x, y):
    """ICC(2,1): two-way random effects, absolute agreement, single measurement (two 'raters')."""
    m = np.column_stack([x, y]).astype(float)
    n, k = m.shape
    grand = m.mean()
    msr = k * np.sum((m.mean(1) - grand) ** 2) / (n - 1)
    msc = n * np.sum((m.mean(0) - grand) ** 2) / (k - 1)
    sse = np.sum((m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + grand) ** 2)
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def bland_altman(reference, test):
    """Bias (mean test-reference), 95% limits of agreement, and proportional-bias slope (diff vs mean)."""
    r, t = np.asarray(reference, float), np.asarray(test, float)
    d, m = t - r, (t + r) / 2
    bias, sd = d.mean(), d.std(ddof=1)
    slope = st.linregress(m, d)
    return {"bias": float(bias), "loa_low": float(bias - 1.96 * sd), "loa_high": float(bias + 1.96 * sd),
            "proportional_bias_slope": float(slope.slope), "proportional_bias_p": float(slope.pvalue),
            "n": int(len(d))}


def regression_agreement(reference, test):
    """OLS test ~ reference: slope and intercept with 95% CI (ideal: slope 1, intercept 0)."""
    res = st.linregress(reference, test)
    tq = st.t.ppf(0.975, len(reference) - 2)
    return {"slope": float(res.slope), "slope_ci": [float(res.slope - tq * res.stderr), float(res.slope + tq * res.stderr)],
            "intercept": float(res.intercept),
            "intercept_ci": [float(res.intercept - tq * res.intercept_stderr), float(res.intercept + tq * res.intercept_stderr)],
            "r2": float(res.rvalue ** 2)}


def agreement_report(reference, test, groups=None):
    """Everything for a 'does the virtual measurement agree with the real one' question."""
    r, t = np.asarray(reference, float), np.asarray(test, float)
    return {"pearson": bootstrap_ci(lambda a, b: np.corrcoef(a, b)[0, 1], r, t, groups=groups),
            "lins_ccc": bootstrap_ci(lins_ccc, r, t, groups=groups),
            "icc_2_1": float(icc_2_1(r, t)),
            "bland_altman": bland_altman(r, t),
            "regression": regression_agreement(r, t)}


# ------------------------------------------------------------------------------------------ classification
def _midrank(x):
    order = np.argsort(x)
    xs = x[order]
    n = len(x)
    ranks = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def delong(y, p1, p2):
    """DeLong test for two correlated ROC AUCs on the same samples. Returns AUCs, difference and p-value."""
    y = np.asarray(y).astype(bool)
    preds = np.vstack([p1, p2]).astype(float)
    pos, neg = preds[:, y], preds[:, ~y]
    m, n = pos.shape[1], neg.shape[1]
    tx = np.array([_midrank(p) for p in pos])
    ty = np.array([_midrank(p) for p in neg])
    tz = np.array([_midrank(p) for p in np.hstack([pos, neg])])
    aucs = (tz[:, :m].sum(1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "diff": float(aucs[0] - aucs[1]),
            "p": float(2 * st.norm.sf(abs(z)))}


def classification_report(y, p, groups=None, n_bins=10):
    """AUC with (grouped) bootstrap CI, Brier score and a reliability curve."""
    y, p = np.asarray(y), np.asarray(p, float)
    if p.min() < 0 or p.max() > 1:  # decision-function scores (e.g. SVM): AUC is valid, Brier/calibration are not
        return {"auc": bootstrap_ci(roc_auc_score, y, p, groups=groups), "brier": None, "calibration": None,
                "note": "scores are not probabilities (decision function): Brier score and calibration not computed",
                "prevalence": float(y.mean()), "n": int(len(y))}
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    curve = [{"bin": b, "mean_pred": float(p[bins == b].mean()), "frac_pos": float(y[bins == b].mean()),
              "n": int((bins == b).sum())} for b in range(n_bins) if (bins == b).any()]
    return {"auc": bootstrap_ci(roc_auc_score, y, p, groups=groups), "brier": float(brier_score_loss(y, p)),
            "prevalence": float(y.mean()), "calibration": curve, "n": int(len(y))}


if __name__ == "__main__":  # self-test on synthetic data with known answers
    rng = np.random.default_rng(1)
    x = rng.normal(100, 20, 300)
    y = x * 1.0 + 5 + rng.normal(0, 3, 300)                      # biased by +5
    rep = agreement_report(x, y)
    assert abs(rep["bland_altman"]["bias"] - 5) < 1 and rep["lins_ccc"]["estimate"] < rep["pearson"]["estimate"]
    lab = rng.integers(0, 2, 400)
    good, bad = lab + rng.normal(0, 0.6, 400), lab + rng.normal(0, 2.0, 400)
    d = delong(lab, good, bad)
    assert d["auc_1"] > d["auc_2"] and d["p"] < 0.05
    assert abs(d["auc_1"] - roc_auc_score(lab, good)) < 1e-9
    print("stats self-test OK", round(rep["bland_altman"]["bias"], 2), round(d["p"], 4))
