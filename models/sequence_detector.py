"""
Sequence-aware detection model
================================
A per-entity chronological window of events is not IID — order matters
(e.g. lateral movement is a *sequence* of resource hops, low-and-slow
exfiltration is a *pattern over days*). We use an LSTM sequence-autoencoder:
it reads a rolling window of the last SEQ_LEN events for an entity and tries
to reconstruct it. High reconstruction error => the recent sequence deviates
from the entity's learned temporal behaviour => higher risk score.

This complements (not replaces) the point-wise baseline_profiler: the final
risk score is a weighted combination of both (see risk_engine.py).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from feature_engineering import FEATURE_COLUMNS

SEQ_LEN = 10


class LSTMAutoEncoder(nn.Module):
    def __init__(self, n_features, hidden_dim=32, latent_dim=12):
        super().__init__()
        self.encoder = nn.LSTM(n_features, hidden_dim, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, n_features, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.encoder(x)          # h: (1, B, hidden)
        z = self.to_latent(h[-1])            # (B, latent)
        h0 = self.from_latent(z).unsqueeze(0)  # (1, B, hidden)
        seq_len = x.size(1)
        repeated = h0.repeat(seq_len, 1, 1).transpose(0, 1)  # (B, seq_len, hidden)
        recon, _ = self.decoder(repeated)
        return recon


def build_sequences(df: pd.DataFrame, feature_columns=FEATURE_COLUMNS, seq_len=SEQ_LEN):
    """Slide a window of seq_len events per entity; pad short histories by
    repeating the first row (keeps cold-start entities usable)."""
    sequences, index_map = [], []
    for entity_id, g in df.groupby("entity_id"):
        g = g.sort_values("timestamp")
        X = g[feature_columns].values
        n = len(X)
        for i in range(n):
            start = max(0, i - seq_len + 1)
            window = X[start:i + 1]
            if len(window) < seq_len:
                pad = np.repeat(window[0:1], seq_len - len(window), axis=0)
                window = np.concatenate([pad, window], axis=0)
            sequences.append(window)
            index_map.append(g.index[i])
    return np.array(sequences, dtype=np.float32), index_map


def normalize_sequences(seqs, stats=None):
    B, T, F = seqs.shape
    flat = seqs.reshape(-1, F)
    if stats is None:
        mean, std = flat.mean(axis=0), flat.std(axis=0) + 1e-6
    else:
        mean, std = stats
    norm = (flat - mean) / std
    norm = np.nan_to_num(norm, nan=0.0, posinf=5.0, neginf=-5.0)
    return norm.reshape(B, T, F), (mean, std)


def train_sequence_model(df, feature_columns=FEATURE_COLUMNS, epochs=15, batch_size=128, lr=1e-3):
    seqs, index_map = build_sequences(df, feature_columns)
    seqs_norm, stats = normalize_sequences(seqs)
    X_t = torch.tensor(seqs_norm, dtype=torch.float32)

    model = LSTMAutoEncoder(n_features=len(feature_columns))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n = X_t.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch = X_t[idx]
            opt.zero_grad()
            recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}/{epochs}  seq_recon_mse={total/n:.4f}")
    return model, stats, index_map


def sequence_scores(model, df, feature_columns=FEATURE_COLUMNS, stats=None):
    seqs, index_map = build_sequences(df, feature_columns)
    seqs_norm, _ = normalize_sequences(seqs, stats)
    X_t = torch.tensor(seqs_norm, dtype=torch.float32)
    with torch.no_grad():
        recon = model(X_t)
        # weight later timesteps more (the "current" event matters most)
        weights = torch.linspace(0.3, 1.0, X_t.size(1)).view(1, -1, 1)
        err = (((recon - X_t) ** 2) * weights).mean(dim=(1, 2)).numpy()
    scores = pd.Series(err, index=index_map).reindex(df.index)
    return scores.values


if __name__ == "__main__":
    from feature_engineering import build_features

    raw = pd.read_csv("../data/synthetic_access_logs.csv")
    feats = build_features(raw)
    print(f"Building sequences for {feats['entity_id'].nunique()} entities...")
    model, stats, _ = train_sequence_model(feats)
    scores = sequence_scores(model, feats, stats=stats)
    feats["seq_score"] = scores
    print(feats[["entity_id", "label", "seq_score"]].sort_values("seq_score", ascending=False).head(15))
