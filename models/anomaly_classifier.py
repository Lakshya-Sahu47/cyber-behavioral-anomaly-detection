"""
Anomaly-type classification
=============================
Once an event is flagged (high risk score from baseline_profiler +
sequence_detector), we still need to tell the analyst WHAT KIND of anomaly it
resembles. We use gradient-boosted trees (LightGBM/XGBoost-style, here via
sklearn's HistGradientBoostingClassifier to keep dependencies light) trained
on the engineered features + the two anomaly scores, against the injected
ground-truth labels.

Why trees, not another deep net: with heavy class imbalance and few labeled
anomalies, tree ensembles are more sample-efficient, and give clean per-feature
importances that feed straight into the explainability layer.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from feature_engineering import FEATURE_COLUMNS

CLASSIFIER_FEATURES = FEATURE_COLUMNS + ["baseline_score", "seq_score"]


def train_classifier(df: pd.DataFrame, label_col="label"):
    """Trains only on labeled anomalies + a sample of 'normal' events
    (class_weight handles the remaining imbalance)."""
    X = df[CLASSIFIER_FEATURES].fillna(0)
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42)

    clf = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.08, max_depth=6,
        class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    print(classification_report(y_test, preds, zero_division=0))
    return clf, (X_test, y_test)


def predict_anomaly_type(clf, df):
    X = df[CLASSIFIER_FEATURES].fillna(0)
    proba = clf.predict_proba(X)
    classes = clf.classes_
    top_idx = np.argmax(proba, axis=1)
    df = df.copy()
    df["predicted_type"] = classes[top_idx]
    df["type_confidence"] = proba[np.arange(len(df)), top_idx]
    return df


if __name__ == "__main__":
    from feature_engineering import build_features
    from baseline_profiler import train_autoencoder, score_baseline
    from sequence_detector import train_sequence_model, sequence_scores

    raw = pd.read_csv("../data/synthetic_access_logs.csv")
    feats = build_features(raw)

    ae_model, ae_stats = train_autoencoder(feats, epochs=10)
    feats["baseline_score"] = score_baseline(feats, ae_model, ae_stats)

    seq_model, seq_stats, _ = train_sequence_model(feats, epochs=5)
    feats["seq_score"] = sequence_scores(seq_model, feats, stats=seq_stats)

    clf, (X_test, y_test) = train_classifier(feats)
    scored = predict_anomaly_type(clf, feats)
    print(scored[["entity_id", "label", "predicted_type", "type_confidence"]].sample(10))
