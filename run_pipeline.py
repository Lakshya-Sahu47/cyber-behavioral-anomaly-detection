"""
End-to-end pipeline: synthetic data -> features -> baseline profiler ->
sequence detector -> anomaly classifier -> risk engine -> explainability ->
scored_alerts.csv (consumed by dashboard/app.py)

Run:
    cd anomaly_detection
    python run_pipeline.py
"""
import sys
import pandas as pd

sys.path.insert(0, "models")
sys.path.insert(0, "data")

from feature_engineering import build_features
from baseline_profiler import train_autoencoder, score_baseline
from sequence_detector import train_sequence_model, sequence_scores
from anomaly_classifier import train_classifier, predict_anomaly_type
from risk_engine import compute_risk_score, alert_queue
from explainability import build_explainer, explain_alerts


def main(csv_path="data/synthetic_access_logs.csv", out_path="reports/scored_alerts.csv"):
    print("1/6 Loading data & engineering features...")
    raw = pd.read_csv(csv_path)
    feats = build_features(raw)

    print("2/6 Training baseline profiler (autoencoder + statistical)...")
    ae_model, ae_stats = train_autoencoder(feats, epochs=20)
    feats["baseline_score"] = score_baseline(feats, ae_model, ae_stats)

    print("3/6 Training sequence detector (LSTM autoencoder)...")
    seq_model, seq_stats, _ = train_sequence_model(feats, epochs=10)
    feats["seq_score"] = sequence_scores(seq_model, feats, stats=seq_stats)

    print("4/6 Training anomaly-type classifier...")
    clf, (X_test, y_test) = train_classifier(feats)
    scored = predict_anomaly_type(clf, feats)

    print("5/6 Computing unified risk score...")
    scored = compute_risk_score(scored)

    print("6/6 Generating SHAP explanations for top alerts (this can be slow, so only for the alert queue)...")
    top_alerts = alert_queue(scored, alert_budget_pct=2.0).copy()
    explainer = build_explainer(clf)
    explained = explain_alerts(clf, explainer, top_alerts)

    explained.to_csv(out_path, index=False)
    scored.to_csv("reports/all_scored_events.csv", index=False)
    print(f"\nSaved {len(explained)} top alerts -> {out_path}")
    print(f"Saved {len(scored)} scored events -> reports/all_scored_events.csv")
    print("\nSample alert:")
    print(explained[["entity_id", "label", "predicted_type", "risk_score", "explanation"]].head(3).to_string())


if __name__ == "__main__":
    main()
