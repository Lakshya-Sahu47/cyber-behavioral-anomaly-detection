"""
Generates two PNGs for the presentation:
  1. architecture_diagram.png  -> Technical Approach slide
  2. dashboard_mockup.png      -> Artifacts slide (static mockup, since no
     headless browser is available in this sandbox to screenshot the live
     Streamlit app — run `streamlit run dashboard/app.py` yourself for a
     real screenshot to swap in).
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd
import numpy as np

# ---------------------------------------------------------------- ARCHITECTURE
fig, ax = plt.subplots(figsize=(13, 6.5))
ax.set_xlim(0, 13)
ax.set_ylim(0, 6.5)
ax.axis("off")

BLUE = "#1F4E79"
LIGHT = "#DCE9F5"
GREEN = "#2E7D32"
AMBER = "#B8860B"

def box(x, y, w, h, text, color=LIGHT, edge=BLUE, fontsize=10, textcolor="#111"):
    b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.12",
                        linewidth=1.6, edgecolor=edge, facecolor=color)
    ax.add_patch(b)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=textcolor, wrap=True, fontweight="medium")
    return b

def arrow(x1, y1, x2, y2, color="#444"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16,
                         linewidth=1.6, color=color)
    ax.add_patch(a)

# Row 1: sources
box(0.3, 5.3, 2.4, 0.9, "Users /\nService Accounts", GREEN + "22", GREEN, 9.5)
box(3.0, 5.3, 2.4, 0.9, "Edge / IoT /\nOT devices", GREEN + "22", GREEN, 9.5)
box(5.7, 5.3, 2.4, 0.9, "Cloud / Network\nAccess logs", GREEN + "22", GREEN, 9.5)
ax.text(4.5, 6.2, "Streaming access & connection events", ha="center", fontsize=11, fontweight="bold")

for x in [1.5, 4.2, 6.9]:
    arrow(x, 5.3, 4.8, 4.7)

# Row 2: ingestion + feature engineering
box(3.3, 3.9, 3.0, 0.8, "Streaming ingestion\n(Kafka / Kinesis)", LIGHT, BLUE)
arrow(4.8, 3.9, 4.8, 3.15)
box(3.0, 2.35, 3.6, 0.8, "Per-entity feature engineering\n(rolling z-scores, geo-velocity,\nresource novelty, fail-rate)", LIGHT, BLUE, 8.5)

# Row 3: three parallel models
arrow(3.4, 2.35, 1.6, 1.55)
arrow(4.8, 2.35, 4.8, 1.55)
arrow(6.2, 2.35, 8.0, 1.55)

box(0.2, 0.75, 2.9, 0.8, "Baseline Profiler\n(autoencoder + stats,\ncold-start blending)", "#FFF3CD", AMBER, 8)
box(3.4, 0.75, 2.9, 0.8, "Sequence Detector\n(LSTM autoencoder\nover event windows)", "#FFF3CD", AMBER, 8)
box(6.6, 0.75, 2.9, 0.8, "Anomaly-type Classifier\n(Gradient boosted trees)", "#FFF3CD", AMBER, 8)

arrow(1.65, 0.75, 4.8, 0.15)
arrow(4.85, 0.75, 4.8, 0.15)
arrow(8.05, 0.75, 4.8, 0.15)

box(9.9, 3.9, 2.9, 0.8, "Risk Engine\n(weighted blend -> 0-100 score)", "#E1D5F5", "#6A1B9A", 8.5)
arrow(9.55, 1.15, 10.9, 3.85)

box(9.9, 2.55, 2.9, 0.8, "Explainability (SHAP)\nfeature attribution", "#E1D5F5", "#6A1B9A", 8.5)
arrow(11.3, 3.9, 11.3, 3.4)

box(9.9, 1.2, 2.9, 0.8, "Analyst Dashboard\nranked alert queue,\nentity drill-down", "#D6EAF8", BLUE, 8)
arrow(11.3, 2.55, 11.3, 2.05)

ax.text(6.5, -0.15, "Fig. 1 — End-to-end pipeline: ingestion -> feature engineering -> "
                    "3 parallel models -> unified risk score -> explainable SOC alert",
        ha="center", fontsize=9, style="italic", color="#555")

plt.tight_layout()
plt.savefig("architecture_diagram.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved architecture_diagram.png")

# ---------------------------------------------------------------- DASHBOARD MOCKUP
np.random.seed(1)
fig2 = plt.figure(figsize=(13, 7.3))
fig2.patch.set_facecolor("#0E1117")

gs = fig2.add_gridspec(3, 3, height_ratios=[0.6, 2.2, 1.6], hspace=0.45, wspace=0.35)

# Title bar
ax_title = fig2.add_subplot(gs[0, :])
ax_title.axis("off")
ax_title.set_facecolor("#0E1117")
ax_title.text(0, 0.5, "🛡️  Behavioral Anomaly Detection — Analyst Console", color="white",
              fontsize=16, fontweight="bold", va="center")

metrics = [("Events processed", "7,285"), ("Alerts (top 2% budget)", "146"),
           ("Entities flagged", "58"), ("Avg. risk score", "78.4")]
for i, (label, val) in enumerate(metrics):
    ax_title.text(0.32 + i * 0.17, 0.5, f"{val}\n", color="#4FC3F7", fontsize=13,
                  fontweight="bold", va="center", ha="center")
    ax_title.text(0.32 + i * 0.17, 0.05, label, color="#AAAAAA", fontsize=8, va="center", ha="center")

# Alert table (mock)
ax_table = fig2.add_subplot(gs[1, :2])
ax_table.axis("off")
ax_table.set_facecolor("#161A23")
cols = ["Entity", "Type", "Risk", "Explanation"]
rows = [
    ["user_00003", "insider_drift", "99.8", "unusual gap since last activity; new resource access"],
    ["edge_device_00071", "device_spoofing", "97.2", "fingerprint mismatch; unfamiliar geo"],
    ["user_00088", "lateral_movement", "95.6", "breadth of new resources; sequence deviation"],
    ["service_account_0021", "credential_stuffing", "93.1", "shared source IP; high failure rate"],
    ["user_00014", "brute_force", "91.4", "auth burst rate; repeated failures"],
]
ax_table.text(0.01, 0.95, "Ranked Alert Queue", color="white", fontsize=12, fontweight="bold")
for c_i, c in enumerate(cols):
    ax_table.text([0.01, 0.22, 0.40, 0.52][c_i], 0.82, c, color="#4FC3F7", fontsize=9, fontweight="bold")
for r_i, row in enumerate(rows):
    y = 0.70 - r_i * 0.13
    for c_i, val in enumerate(row):
        ax_table.text([0.01, 0.22, 0.40, 0.52][c_i], y, val, color="#E0E0E0", fontsize=8.3,
                       wrap=True)
ax_table.set_xlim(0, 1)
ax_table.set_ylim(0, 1)

# Pie chart (mock)
ax_pie = fig2.add_subplot(gs[1, 2])
ax_pie.set_facecolor("#161A23")
labels = ["brute_force", "lateral_movement", "credential_stuffing", "low_and_slow", "device_spoofing", "other"]
sizes = [40, 20, 14, 12, 8, 6]
colors = ["#4FC3F7", "#81C784", "#FFB74D", "#E57373", "#BA68C8", "#90A4AE"]
ax_pie.pie(sizes, labels=None, colors=colors, startangle=90,
           wedgeprops=dict(width=0.45, edgecolor="#161A23"))
ax_pie.legend(labels, loc="center", fontsize=6.5, labelcolor="white", frameon=False, ncol=1)
ax_pie.set_title("Alerts by predicted type", color="white", fontsize=9)

# Entity drilldown line chart (mock)
ax_line = fig2.add_subplot(gs[2, :2])
ax_line.set_facecolor("#161A23")
t = np.arange(20)
risk = np.clip(30 + np.cumsum(np.random.randn(20) * 5) + (t > 14) * 40, 0, 100)
ax_line.plot(t, risk, color="#4FC3F7", marker="o", markersize=3)
ax_line.fill_between(t, risk, color="#4FC3F7", alpha=0.15)
ax_line.set_title("Risk score over time — user_00003", color="white", fontsize=9)
ax_line.tick_params(colors="#AAAAAA", labelsize=7)
for spine in ax_line.spines.values():
    spine.set_color("#333")

# Entity history mini-table
ax_hist = fig2.add_subplot(gs[2, 2])
ax_hist.axis("off")
ax_hist.set_facecolor("#161A23")
ax_hist.text(0.02, 0.95, "Recent raw events", color="white", fontsize=9, fontweight="bold")
hist_rows = ["file_share_9 · token · 812s", "/svc/14 · token · 44s (NEW)", "port_3389 · token · 12s (NEW)"]
for i, r in enumerate(hist_rows):
    ax_hist.text(0.02, 0.75 - i * 0.18, r, color="#E0E0E0", fontsize=7.5)
ax_hist.set_xlim(0, 1)
ax_hist.set_ylim(0, 1)

for ax in [ax_table, ax_pie, ax_line, ax_hist]:
    ax.set_facecolor("#161A23")

plt.savefig("dashboard_mockup.png", dpi=180, bbox_inches="tight", facecolor="#0E1117")
plt.close()
print("Saved dashboard_mockup.png")
