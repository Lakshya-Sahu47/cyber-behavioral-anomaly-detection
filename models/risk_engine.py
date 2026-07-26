"""
Risk scoring engine — combines everything into one 0-100 analyst-facing score.

risk_score = 100 * ( w1 * baseline_score_percentile
                    + w2 * seq_score_percentile
                    + w3 * classifier_confidence_if_not_normal )

Weights are configurable; defaults favor the sequence model slightly since
most attack patterns in the taxonomy (lateral movement, low-and-slow,
credential stuffing) are sequence/frequency phenomena rather than single-event
outliers.
"""
import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {"baseline": 0.35, "sequence": 0.45, "classifier": 0.20}


def compute_risk_score(df: pd.DataFrame, weights=None) -> pd.DataFrame:
    weights = weights or DEFAULT_WEIGHTS
    df = df.copy()
    base_pct = df["baseline_score"].rank(pct=True)
    seq_pct = df["seq_score"].rank(pct=True)
    clf_component = np.where(df["predicted_type"] != "normal", df["type_confidence"], 0)

    df["risk_score"] = 100 * (
        weights["baseline"] * base_pct +
        weights["sequence"] * seq_pct +
        weights["classifier"] * clf_component
    )
    return df


def alert_queue(df: pd.DataFrame, alert_budget_pct=1.0) -> pd.DataFrame:
    """Return the top X% highest-risk events — mirrors a realistic SOC
    analyst alert budget (evaluation criterion: FP rate at top 1%)."""
    threshold = df["risk_score"].quantile(1 - alert_budget_pct / 100)
    return df[df["risk_score"] >= threshold].sort_values("risk_score", ascending=False)
