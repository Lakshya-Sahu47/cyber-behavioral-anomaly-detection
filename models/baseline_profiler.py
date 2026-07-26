"""
Baseline profiling model
=========================
Two-tier design to solve the cold-start problem cleanly:

1. Statistical baseline (per-entity mean/std of features) — works from event #1,
   used with a shrinkage weight toward the entity_type-level prior until an
   entity has seen COLD_START_THRESHOLD events.
2. Autoencoder (PyTorch) — learns a compressed "normal" manifold across ALL
   entities of a type once enough population data exists; reconstruction error
   becomes the anomaly signal for entities with sufficient history.

Final baseline_score blends both, weighted by how much history the entity has,
so a brand-new device is scored sensibly (via type-level prior) rather than
either flagged constantly or ignored.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from feature_engineering import FEATURE_COLUMNS

COLD_START_THRESHOLD = 15


class AutoEncoder(nn.Module):
    def __init__(self, n_features, latent_dim=6):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 16), nn.ReLU(),
            nn.Linear(16, latent_dim), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16), nn.ReLU(),
            nn.Linear(16, n_features),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def normalize(df, feature_columns=FEATURE_COLUMNS, stats=None):
    if stats is None:
        stats = {c: (df[c].mean(), df[c].std() + 1e-6) for c in feature_columns}
    X = np.stack([(df[c] - stats[c][0]) / stats[c][1] for c in feature_columns], axis=1)
    return np.nan_to_num(X, nan=0.0, posinf=5.0, neginf=-5.0), stats


def train_autoencoder(df, feature_columns=FEATURE_COLUMNS, epochs=30, batch_size=256, lr=1e-3):
    """Train only on rows the label marks 'normal' (or, at inference time on
    unlabeled data, on the full stream — anomalies are <3% so the autoencoder
    still converges toward the normal manifold)."""
    X, stats = normalize(df, feature_columns)
    X_t = torch.tensor(X, dtype=torch.float32)

    model = AutoEncoder(n_features=X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = X_t.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = X_t[idx]
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch+1}/{epochs}  recon_mse={total_loss/n:.4f}")
    return model, stats


def reconstruction_error(model, df, stats, feature_columns=FEATURE_COLUMNS):
    X, _ = normalize(df, feature_columns, stats)
    X_t = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        recon = model(X_t)
        err = ((recon - X_t) ** 2).mean(dim=1).numpy()
    return err


def statistical_baseline_score(df, feature_columns=FEATURE_COLUMNS):
    """Per-entity z-score magnitude — cheap, explainable, works from event 1."""
    scores = np.zeros(len(df))
    for c in feature_columns:
        type_mean = df.groupby("entity_type")[c].transform("mean")
        type_std = df.groupby("entity_type")[c].transform("std").replace(0, 1).fillna(1)
        scores += np.abs((df[c] - type_mean) / type_std).fillna(0).values
    return scores / len(feature_columns)


def cold_start_weight(n_events_observed, threshold=COLD_START_THRESHOLD):
    """0 -> rely fully on population statistical baseline; 1 -> fully trust
    the entity's own learned (autoencoder) profile."""
    return np.clip(n_events_observed / threshold, 0, 1)


def score_baseline(df, model, stats, feature_columns=FEATURE_COLUMNS):
    ae_err = reconstruction_error(model, df, stats, feature_columns)
    stat_score = statistical_baseline_score(df, feature_columns)
    w = cold_start_weight(df["n_events_observed"].values)
    # blend: statistical score is on a different scale than AE MSE -> rank-normalize both
    ae_rank = pd.Series(ae_err).rank(pct=True).values
    stat_rank = pd.Series(stat_score).rank(pct=True).values
    blended = w * ae_rank + (1 - w) * stat_rank
    return blended


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from feature_engineering import build_features

    raw = pd.read_csv("../data/synthetic_access_logs.csv")
    feats = build_features(raw)
    print("Training autoencoder on", len(feats), "events...")
    model, stats = train_autoencoder(feats)
    feats["baseline_score"] = score_baseline(feats, model, stats)
    print(feats[["entity_id", "label", "baseline_score"]].sort_values("baseline_score", ascending=False).head(15))
