"""
Canonical evaluation script for the paper:
"The Economic Value of Textual Information in Credit Risk Prediction:
 Evidence from a Machine Learning Framework" (Dongwoo Kim, Baekseok University)

Implements the paper's evaluation protocol exactly:
- Repeated stratified 5-fold cross-validation with 5 repeats
  (RepeatedStratifiedKFold; n_splits=5, n_repeats=5, random_state=42 -> 25 test folds)
- Six classifiers: LR, DT, NB, RF, GB, XGB
- Feature settings: structured-only (Phase 0), text-only TF-IDF (Phase 1, Stage 1),
  merged structured+TF-IDF (Phase 2, Stage 1)
- Metrics: ROC-AUC, PR-AUC, H-measure, Recall and F1 at a fixed threshold (tau = 0.5)
- All preprocessing (scaling, TF-IDF fitting) is performed inside each training fold
  to prevent information leakage; results are reported as mean with 95% CIs.

Usage:
    python run_repeated_cv_benchmark.py --setting structured
    python run_repeated_cv_benchmark.py --setting text
    python run_repeated_cv_benchmark.py --setting merged
"""

import sys, argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

import config
from utils.data_loader import load_raw_data, encode_target


# ----------------------------------------------------------------------------
# H-measure (Hand, 2009), Beta(2,2) severity distribution
# ----------------------------------------------------------------------------
def h_measure(y_true, y_score):
    """Hand's H-measure (Hand, 2009) with a symmetric Beta(2,2) cost distribution.

    L(c; t) = c*pi1*(1 - F1(t)) + (1-c)*pi0*F0(t) is minimised over thresholds t
    for each cost c, weighted by w(c) = Beta(2,2); H = 1 - E[min L] / E[min L_trivial].
    """
    from scipy.stats import beta as beta_dist
    y = np.asarray(y_true).astype(int)
    s = np.asarray(y_score, dtype=float)
    n = len(y); n1 = int(y.sum()); n0 = n - n1
    pi0, pi1 = n0 / n, n1 / n

    order = np.argsort(-s, kind="mergesort")
    ys = y[order]; ss = s[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    last = np.r_[np.where(np.diff(ss))[0], n - 1]
    F1 = np.r_[0.0, tp[last] / n1]   # TPR
    F0 = np.r_[0.0, fp[last] / n0]   # FPR

    # upper convex hull of ROC points (including (0,0) and (1,1))
    hull = [(0.0, 0.0)]
    for x, yv in zip(F0[1:], F1[1:]):
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (y2 - y1) * (x - x1) <= (yv - y1) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append((x, yv))
    hx = np.array([q[0] for q in hull]); hy = np.array([q[1] for q in hull])
    m = len(hull)

    # cost values at which the optimal hull vertex switches:
    # indifference between consecutive vertices: c*pi1*dF1 = (1-c)*pi0*dF0
    c = [0.0]
    for i in range(1, m):
        dF0 = hx[i] - hx[i - 1]; dF1 = hy[i] - hy[i - 1]
        denom = pi0 * dF0 + pi1 * dF1
        c.append((pi0 * dF0) / denom if denom > 0 else 1.0)
    c.append(1.0)

    a, b = 2.0, 2.0
    def int_c_w(lo, hi):    # integral of c*w(c) over [lo, hi]
        return (a / (a + b)) * (beta_dist.cdf(hi, a + 1, b) - beta_dist.cdf(lo, a + 1, b))
    def int_1mc_w(lo, hi):  # integral of (1-c)*w(c) over [lo, hi]
        return (b / (a + b)) * (beta_dist.cdf(hi, a, b + 1) - beta_dist.cdf(lo, a, b + 1))

    L = 0.0
    for i in range(m):
        lo, hi = c[i], c[i + 1]
        if hi > lo:
            L += pi1 * (1 - hy[i]) * int_c_w(lo, hi) + pi0 * hx[i] * int_1mc_w(lo, hi)

    # reference loss: best trivial classifier (all-negative vs all-positive)
    cswitch = pi0  # c*pi1 = (1-c)*pi0  ->  c = pi0/(pi0+pi1) = pi0
    Lmax = pi1 * int_c_w(0.0, cswitch) + pi0 * int_1mc_w(cswitch, 1.0)
    return 1.0 - L / Lmax if Lmax > 0 else 0.0


def get_models(seed=42):
    return {
        "LR": LogisticRegression(max_iter=1000, random_state=seed),
        "DT": DecisionTreeClassifier(random_state=seed),
        "NB": GaussianNB(),
        "RF": RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1),
        "GB": GradientBoostingClassifier(n_estimators=100, random_state=seed),
        "XGB": XGBClassifier(n_estimators=100, random_state=seed, eval_metric="logloss", n_jobs=-1),
    }


def build_features(df, setting, train_idx, test_idx):
    """Fold-safe feature construction (fit on train only)."""
    Xs = df[config.STRUCTURED_FEATURES].astype(float).values
    scaler = StandardScaler().fit(Xs[train_idx])
    Xs_tr, Xs_te = scaler.transform(Xs[train_idx]), scaler.transform(Xs[test_idx])
    if setting == "structured":
        return Xs_tr, Xs_te
    text = (df[config.TEXT_COLUMNS[0]].fillna("") + " " +
            df[config.TEXT_COLUMNS[1]].fillna("") + " " +
            df[config.TEXT_COLUMNS[2]].fillna("")).values
    vec = TfidfVectorizer(max_features=config.TFIDF_MAX_FEATURES,
                          min_df=config.TFIDF_MIN_DF, max_df=config.TFIDF_MAX_DF,
                          ngram_range=config.TFIDF_NGRAM_RANGE)
    Xt_tr = vec.fit_transform(text[train_idx]).toarray()
    Xt_te = vec.transform(text[test_idx]).toarray()
    if setting == "text":
        return Xt_tr, Xt_te
    return np.hstack([Xs_tr, Xt_tr]), np.hstack([Xs_te, Xt_te])


def ci95(x):
    x = np.asarray(x, dtype=float)
    m = x.mean(); h = 1.96 * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["structured", "text", "merged"], default="structured")
    args = ap.parse_args()

    df = load_raw_data()
    df = encode_target(df)
    y = df["target"].values

    rskf = RepeatedStratifiedKFold(n_splits=config.CV_FOLDS,
                                   n_repeats=config.CV_REPEATS,
                                   random_state=config.RANDOM_SEED)
    records = []
    for fold, (tr, te) in enumerate(rskf.split(np.zeros(len(y)), y)):
        Xtr, Xte = build_features(df, args.setting, tr, te)
        for name, model in get_models(config.RANDOM_SEED).items():
            model.fit(Xtr, y[tr])
            p = model.predict_proba(Xte)[:, 1]
            yhat = (p >= 0.5).astype(int)
            records.append({
                "fold": fold, "model": name,
                "roc_auc": roc_auc_score(y[te], p),
                "pr_auc": average_precision_score(y[te], p),
                "h_measure": h_measure(y[te], p),
                "recall": recall_score(y[te], yhat),
                "f1": f1_score(y[te], yhat),
            })
    res = pd.DataFrame(records)
    print(f"\n=== Setting: {args.setting} | folds: {res['fold'].nunique()} ===")
    for name, g in res.groupby("model"):
        row = {m: ci95(g[m].values) for m in ["roc_auc", "pr_auc", "h_measure", "recall", "f1"]}
        line = " | ".join(f"{m}: {v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]" for m, v in row.items())
        print(f"{name:>4} | {line}")
    out = config.get_table_path(f"repeated_cv_{args.setting}")
    res.to_csv(out, index=False)
    print(f"\nPer-fold results saved to {out}")


if __name__ == "__main__":
    main()
