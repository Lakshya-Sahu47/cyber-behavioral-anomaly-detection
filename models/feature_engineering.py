"""
Feature engineering: turns raw access-log rows into numeric features
used by both the baseline profiler and the sequence detector.

Design choice: features are computed relative to each entity's OWN rolling
history (not global averages), so the model naturally adapts to per-entity
"normal" and handles concept drift via a rolling window.
"""
import json
from math import radians, sin, cos, sqrt, atan2

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


def parse_geo(x):
    d = json.loads(x)
    return d["lat"], d["lon"]


def build_features(df: pd.DataFrame, rolling_window: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)
    df["hour"] = df["timestamp"].dt.hour
    df["is_off_hours"] = ((df["hour"] < 6) | (df["hour"] > 22)).astype(int)
    df[["lat", "lon"]] = df["geo_location"].apply(lambda x: pd.Series(parse_geo(x)))
    df["cmd_count"] = df["command_sequence"].apply(lambda x: len(json.loads(x)))
    df["auth_fail"] = (df.get("auth_result", "success") == "fail").astype(int)

    feats = []
    for entity_id, g in df.groupby("entity_id"):
        g = g.reset_index(drop=True)
        prev_ip = g["source_ip"].shift(1)
        prev_lat, prev_lon = g["lat"].shift(1), g["lon"].shift(1)
        prev_ts = g["timestamp"].shift(1)
        prev_fp = g["device_fingerprint"].shift(1)

        g["ip_changed"] = (g["source_ip"] != prev_ip).astype(float)
        g["time_since_last_s"] = (g["timestamp"] - prev_ts).dt.total_seconds().fillna(3600)
        g["geo_velocity_kmh"] = [
            haversine_km(a, b, c, d) / max(t / 3600, 1e-3) if not np.isnan(a) else 0.0
            for a, b, c, d, t in zip(prev_lat, prev_lon, g["lat"], g["lon"], g["time_since_last_s"])
        ]
        g["fingerprint_changed"] = (g["device_fingerprint"] != prev_fp).astype(float).fillna(0)

        # rolling per-entity baseline (expanding at cold-start, then windowed)
        roll_hour = g["hour"].rolling(window=rolling_window, min_periods=1)
        g["hist_hour_mean"] = roll_hour.mean().shift(1)
        g["hist_hour_std"] = roll_hour.std().shift(1).fillna(0)
        g["hour_zscore"] = ((g["hour"] - g["hist_hour_mean"]) / g["hist_hour_std"].replace(0, 1)).fillna(0)

        g["new_resource"] = (~g["resource_accessed"].duplicated()).astype(int)

        g["fail_rate_5"] = g["auth_fail"].rolling(5, min_periods=1).mean()
        g["events_last_1min"] = g["time_since_last_s"].rolling(5, min_periods=1).apply(
            lambda x: np.sum(x < 60), raw=True)

        g["n_events_observed"] = np.arange(1, len(g) + 1)  # for cold-start weighting
        feats.append(g)

    out = pd.concat(feats, ignore_index=True)
    return out


FEATURE_COLUMNS = [
    "hour_zscore", "is_off_hours", "ip_changed", "geo_velocity_kmh",
    "fingerprint_changed", "new_resource", "fail_rate_5", "events_last_1min",
    "session_duration", "cmd_count", "time_since_last_s",
]
