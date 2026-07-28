"""
SHAP attribution analysis for the paper (Section 2.9 / Table 11 / Fig 2).

Computes TreeSHAP values for the tuned XGB model in the merged
structured + TF-IDF setting, strictly on held-out test folds under the same
repeated stratified 5-fold cross-validation protocol (5 repeats) used for the
performance evaluation. Global importance is summarised by mean(|SHAP|)
aggregated over out-of-fold predictions across all repeats, grouped into
structured variables (13) versus TF-IDF text features (100).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier

import config
from utils.data_loader import load_raw_data, encode_target
from run_repeated_cv_benchmark import build_features


def main():
    import shap  # TreeSHAP

    df = load_raw_data()
    df = encode_target(df)
    y = df["target"].values
    n_struct = len(config.STRUCTURED_FEATURES)

    rskf = RepeatedStratifiedKFold(n_splits=config.CV_FOLDS,
                                   n_repeats=config.CV_REPEATS,
                                   random_state=config.RANDOM_SEED)
    abs_shap_sum = None
    n_rows = 0
    for tr, te in rskf.split(np.zeros(len(y)), y):
        Xtr, Xte = build_features(df, "merged", tr, te)
        model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                              random_state=config.RANDOM_SEED,
                              eval_metric="logloss", n_jobs=-1)
        model.fit(Xtr, y[tr])
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(Xte)
        a = np.abs(sv)
        abs_shap_sum = a.sum(axis=0) if abs_shap_sum is None else abs_shap_sum + a.sum(axis=0)
        n_rows += len(te)

    mean_abs = abs_shap_sum / n_rows
    struct_share = mean_abs[:n_struct].sum()
    text_share = mean_abs[n_struct:].sum()
    total = struct_share + text_share
    print(f"Structured group mean(|SHAP|) sum: {struct_share:.4f} ({100*struct_share/total:.1f}%)")
    print(f"Text (TF-IDF) group mean(|SHAP|) sum: {text_share:.4f} ({100*text_share/total:.1f}%)")
    out = config.get_table_path("shap_group_contribution")
    pd.DataFrame({
        "group": ["structured", "text_tfidf"],
        "sum_mean_abs_shap": [struct_share, text_share],
        "share": [struct_share/total, text_share/total],
    }).to_csv(out, index=False)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
