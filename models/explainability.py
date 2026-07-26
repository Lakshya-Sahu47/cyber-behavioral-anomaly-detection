"""
Explainability layer
======================
SOC analysts need "why", not just a number. For every alert we compute
SHAP values against the anomaly classifier (tree model -> fast TreeExplainer)
and turn the top contributing features into a human-readable reason string,
e.g. "flagged due to geo-velocity spike (+0.31) and new device fingerprint (+0.18)".
"""
import shap
import numpy as np
import pandas as pd

from anomaly_classifier import CLASSIFIER_FEATURES

READABLE_NAMES = {
    "hour_zscore": "unusual access hour",
    "is_off_hours": "off-hours access",
    "ip_changed": "source IP changed",
    "geo_velocity_kmh": "geo-velocity spike (impossible travel)",
    "fingerprint_changed": "device fingerprint mismatch",
    "new_resource": "access to a never-before-seen resource",
    "fail_rate_5": "elevated authentication failure rate",
    "events_last_1min": "high event burst rate",
    "session_duration": "atypical session duration",
    "cmd_count": "unusual command sequence length",
    "time_since_last_s": "unusual gap since last activity",
    "baseline_score": "deviation from entity's historical baseline",
    "seq_score": "deviation from entity's typical event sequence",
}


def build_explainer(clf):
    return shap.TreeExplainer(clf)


def explain_alerts(clf, explainer, df: pd.DataFrame, top_k=3) -> pd.DataFrame:
    X = df[CLASSIFIER_FEATURES].fillna(0)
    shap_values = explainer.shap_values(X)  # list per class, or array for binary

    # For multi-class HistGradientBoostingClassifier, shap_values may be
    # (n_samples, n_features, n_classes) depending on shap version.
    sv = np.array(shap_values)
    if sv.ndim == 3:
        # pick the contribution toward the PREDICTED class for each row
        pred_idx = [list(clf.classes_).index(c) for c in df["predicted_type"]]
        n_classes = len(clf.classes_)
        n_samples = len(df)
        if sv.shape[0] == n_classes:
            # Shape is (n_classes, n_samples, n_features)
            row_shap = np.stack([sv[pred_idx[i], i, :] for i in range(n_samples)])
        else:
            # Shape is (n_samples, n_features, n_classes)
            row_shap = np.stack([sv[i, :, pred_idx[i]] for i in range(n_samples)])
    else:
        row_shap = sv

    reasons = []
    for i in range(len(df)):
        contribs = row_shap[i]
        top_idx = np.argsort(-np.abs(contribs))[:top_k]
        parts = []
        for idx in top_idx:
            feat = CLASSIFIER_FEATURES[idx]
            sign = "+" if contribs[idx] >= 0 else "-"
            parts.append(f"{READABLE_NAMES.get(feat, feat)} ({sign}{abs(contribs[idx]):.2f})")
        reasons.append("Flagged due to: " + "; ".join(parts))

    df = df.copy()
    df["explanation"] = reasons
    return df


if __name__ == "__main__":
    print("See dashboard/app.py or the notebook pipeline for an end-to-end example:")
    print("  clf, explainer = build_explainer(clf)")
    print("  explained = explain_alerts(clf, explainer, alerts_df)")
