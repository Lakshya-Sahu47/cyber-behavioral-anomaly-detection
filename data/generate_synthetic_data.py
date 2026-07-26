"""
Synthetic Access-Log Generator for Behavioral Anomaly Detection
================================================================
Generates per-entity (user / service_account / edge_device) "normal" behavioural
baselines, then injects labeled attack patterns at controlled rates.

Schema matches the problem statement:
entity_id, entity_type, timestamp, source_ip, geo_location, resource_accessed,
auth_method, session_duration, command_sequence, device_fingerprint, label

Usage:
    python generate_synthetic_data.py --n_entities 500 --days 30 --anomaly_rate 0.02
"""

import argparse
import json
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

ENTITY_TYPES = ["user", "service_account", "edge_device"]
AUTH_METHODS = ["password", "token", "certificate", "biometric"]
RESOURCE_POOL = [f"/svc/{i}" for i in range(1, 40)] + [f"file_share_{i}" for i in range(1, 15)] + \
                [f"port_{p}" for p in [22, 443, 3389, 502, 8080, 8443]]
OS_FW_VERSIONS = ["Win10-19045", "Ubuntu-22.04", "FW-2.3.1", "FW-3.0.0", "iOS-17.2", "RTOS-1.8"]

ANOMALY_TYPES = [
    "brute_force",
    "impossible_travel",
    "credential_stuffing",
    "lateral_movement",
    "device_spoofing",
    "low_and_slow_exfil",
    "insider_drift",  # ambiguous edge-case, used for FP tuning
]


def random_geo():
    return {"lat": round(float(fake.latitude()), 4), "lon": round(float(fake.longitude()), 4),
            "city": fake.city(), "country": fake.country_code()}


def build_entity_profiles(n_entities):
    """Create a stable 'habitual' behavioural profile per entity."""
    profiles = []
    for i in range(n_entities):
        entity_type = np.random.choice(ENTITY_TYPES, p=[0.6, 0.25, 0.15])
        home_geo = random_geo()
        profile = {
            "entity_id": f"{entity_type}_{i:05d}",
            "entity_type": entity_type,
            "home_geo": home_geo,
            "typical_hours": sorted(np.random.choice(range(24), size=np.random.randint(2, 5), replace=False)),
            "typical_resources": list(np.random.choice(RESOURCE_POOL, size=np.random.randint(3, 8), replace=False)),
            "typical_auth": np.random.choice(AUTH_METHODS),
            "device_fingerprint": np.random.choice(OS_FW_VERSIONS),
            "source_ip": fake.ipv4(),
            "avg_session_s": np.random.randint(60, 1800),
        }
        profiles.append(profile)
    return profiles


def normal_event(profile, ts):
    hour = np.random.choice(profile["typical_hours"])
    ts = ts.replace(hour=int(hour), minute=np.random.randint(0, 59))
    resource = np.random.choice(profile["typical_resources"])
    session_dur = max(5, int(np.random.normal(profile["avg_session_s"], profile["avg_session_s"] * 0.15)))
    cmds = list(np.random.choice(["read", "write", "list", "exec", "connect"],
                                  size=np.random.randint(1, 4)))
    return {
        "entity_id": profile["entity_id"],
        "entity_type": profile["entity_type"],
        "timestamp": ts.isoformat(),
        "source_ip": profile["source_ip"],
        "geo_location": json.dumps(profile["home_geo"]),
        "resource_accessed": resource,
        "auth_method": profile["typical_auth"],
        "session_duration": session_dur,
        "command_sequence": json.dumps(cmds),
        "device_fingerprint": profile["device_fingerprint"],
        "auth_result": "success",
        "label": "normal",
    }


def inject_brute_force(profile, ts, rows):
    attacker_ip = fake.ipv4()
    n = np.random.randint(15, 60)
    for k in range(n):
        rows.append({
            "entity_id": profile["entity_id"], "entity_type": profile["entity_type"],
            "timestamp": (ts + timedelta(seconds=k * np.random.randint(1, 5))).isoformat(),
            "source_ip": attacker_ip, "geo_location": json.dumps(random_geo()),
            "resource_accessed": profile["typical_resources"][0], "auth_method": "password",
            "session_duration": 0, "command_sequence": json.dumps(["auth_fail"]),
            "device_fingerprint": "unknown", "auth_result": "fail" if k < n - 1 else "success",
            "label": "brute_force",
        })


def inject_impossible_travel(profile, ts, rows):
    geo_a = profile["home_geo"]
    geo_b = random_geo()
    rows.append({**normal_event(profile, ts), "label": "impossible_travel"})
    rows.append({
        "entity_id": profile["entity_id"], "entity_type": profile["entity_type"],
        "timestamp": (ts + timedelta(minutes=np.random.randint(5, 20))).isoformat(),
        "source_ip": fake.ipv4(), "geo_location": json.dumps(geo_b),
        "resource_accessed": np.random.choice(profile["typical_resources"]),
        "auth_method": profile["typical_auth"], "session_duration": np.random.randint(60, 600),
        "command_sequence": json.dumps(["read"]), "device_fingerprint": profile["device_fingerprint"],
        "auth_result": "success", "label": "impossible_travel",
    })


def inject_credential_stuffing(profiles, ts, rows):
    few_ips = [fake.ipv4() for _ in range(np.random.randint(1, 3))]
    victims = np.random.choice(profiles, size=min(len(profiles), np.random.randint(20, 50)), replace=False)
    for v in victims:
        rows.append({
            "entity_id": v["entity_id"], "entity_type": v["entity_type"],
            "timestamp": (ts + timedelta(seconds=np.random.randint(0, 120))).isoformat(),
            "source_ip": np.random.choice(few_ips), "geo_location": json.dumps(random_geo()),
            "resource_accessed": "login_portal", "auth_method": "password",
            "session_duration": 0, "command_sequence": json.dumps(["auth_fail"]),
            "device_fingerprint": "unknown", "auth_result": "fail",
            "label": "credential_stuffing",
        })


def inject_lateral_movement(profile, ts, rows):
    unusual_resources = [r for r in RESOURCE_POOL if r not in profile["typical_resources"]]
    breadth = np.random.choice(unusual_resources, size=min(len(unusual_resources), np.random.randint(6, 12)), replace=False)
    for i, r in enumerate(breadth):
        rows.append({
            "entity_id": profile["entity_id"], "entity_type": profile["entity_type"],
            "timestamp": (ts + timedelta(minutes=i * 2)).isoformat(),
            "source_ip": profile["source_ip"], "geo_location": json.dumps(profile["home_geo"]),
            "resource_accessed": r, "auth_method": profile["typical_auth"],
            "session_duration": np.random.randint(10, 120),
            "command_sequence": json.dumps(["list", "read", "exec"]),
            "device_fingerprint": profile["device_fingerprint"], "auth_result": "success",
            "label": "lateral_movement",
        })


def inject_device_spoofing(profile, ts, rows):
    spoofed_fp = np.random.choice([f for f in OS_FW_VERSIONS if f != profile["device_fingerprint"]])
    rows.append({
        "entity_id": profile["entity_id"], "entity_type": profile["entity_type"],
        "timestamp": ts.isoformat(), "source_ip": fake.ipv4(),
        "geo_location": json.dumps(random_geo()),
        "resource_accessed": np.random.choice(profile["typical_resources"]),
        "auth_method": profile["typical_auth"], "session_duration": np.random.randint(30, 300),
        "command_sequence": json.dumps(["connect"]), "device_fingerprint": spoofed_fp,
        "auth_result": "success", "label": "device_spoofing",
    })


def inject_low_and_slow(profile, ts, rows, days=10):
    for d in range(days):
        day_ts = ts + timedelta(days=d, hours=np.random.randint(1, 4))  # off-hours
        rows.append({
            "entity_id": profile["entity_id"], "entity_type": profile["entity_type"],
            "timestamp": day_ts.isoformat(), "source_ip": profile["source_ip"],
            "geo_location": json.dumps(profile["home_geo"]),
            "resource_accessed": np.random.choice(RESOURCE_POOL),
            "auth_method": profile["typical_auth"], "session_duration": np.random.randint(5, 30),
            "command_sequence": json.dumps(["read", "copy"]),
            "device_fingerprint": profile["device_fingerprint"], "auth_result": "success",
            "label": "low_and_slow_exfil",
        })


def inject_insider_drift(profile, ts, rows, weeks=4):
    footprint = list(profile["typical_resources"])
    for w in range(weeks):
        new_r = np.random.choice([r for r in RESOURCE_POOL if r not in footprint])
        footprint.append(new_r)
        rows.append({
            "entity_id": profile["entity_id"], "entity_type": profile["entity_type"],
            "timestamp": (ts + timedelta(weeks=w)).isoformat(), "source_ip": profile["source_ip"],
            "geo_location": json.dumps(profile["home_geo"]), "resource_accessed": new_r,
            "auth_method": profile["typical_auth"], "session_duration": np.random.randint(60, 300),
            "command_sequence": json.dumps(["read", "write"]),
            "device_fingerprint": profile["device_fingerprint"], "auth_result": "success",
            "label": "insider_drift",
        })


INJECTORS = {
    "brute_force": inject_brute_force,
    "impossible_travel": inject_impossible_travel,
    "device_spoofing": inject_device_spoofing,
    "lateral_movement": inject_lateral_movement,
    "low_and_slow_exfil": inject_low_and_slow,
    "insider_drift": inject_insider_drift,
}


def generate(n_entities=500, days=30, events_per_entity_per_day=3, anomaly_rate=0.02, out_path="synthetic_access_logs.csv"):
    profiles = build_entity_profiles(n_entities)
    rows = []
    start = datetime(2026, 6, 1)

    for profile in profiles:
        for d in range(days):
            for _ in range(np.random.poisson(events_per_entity_per_day)):
                ts = start + timedelta(days=d, hours=np.random.randint(0, 24), minutes=np.random.randint(0, 59))
                rows.append(normal_event(profile, ts))

    # Inject per-entity anomaly patterns
    n_anom_entities = int(n_entities * anomaly_rate * 10)  # spread across the period
    chosen = np.random.choice(profiles, size=min(n_anom_entities, len(profiles)), replace=False)
    for profile in chosen:
        atype = np.random.choice(list(INJECTORS.keys()))
        ts = start + timedelta(days=np.random.randint(0, days), hours=np.random.randint(0, 24))
        INJECTORS[atype](profile, ts, rows)

    # A few credential-stuffing waves (multi-entity, few-IP pattern)
    for _ in range(max(1, days // 10)):
        ts = start + timedelta(days=np.random.randint(0, days))
        inject_credential_stuffing(profiles, ts, rows)

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df.to_csv(out_path, index=False)

    ground_truth = df[["entity_id", "timestamp", "label"]].copy()
    ground_truth.to_csv(out_path.replace(".csv", "_labels.csv"), index=False)

    # Inference-time feed hides the label
    inference_df = df.drop(columns=["label"])
    inference_df.to_csv(out_path.replace(".csv", "_unlabeled.csv"), index=False)

    print(f"Generated {len(df)} events across {n_entities} entities.")
    print(df["label"].value_counts())
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_entities", type=int, default=500)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--events_per_day", type=int, default=3)
    p.add_argument("--anomaly_rate", type=float, default=0.02)
    p.add_argument("--out", type=str, default="synthetic_access_logs.csv")
    args = p.parse_args()
    generate(args.n_entities, args.days, args.events_per_day, args.anomaly_rate, args.out)
