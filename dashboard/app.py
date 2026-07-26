"""
SOC Analyst Dashboard (Streamlit)
===================================
Consumes reports/scored_alerts.csv and reports/all_scored_events.csv
(produced by run_pipeline.py) and renders:
  - Ranked alert queue with risk score & predicted anomaly type
  - Contributing-factor explanation per alert (from SHAP)
  - Entity history view (drill-down into an entity's recent events)

Run:
    cd anomaly_detection
    streamlit run dashboard/app.py
"""
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="Behavioral Anomaly Detection — SOC Dashboard", layout="wide")

@st.cache_data
def load_data():
    alerts = pd.read_csv("reports/scored_alerts.csv")
    all_events = pd.read_csv("reports/all_scored_events.csv")
    return alerts, all_events

alerts, all_events = load_data()

st.title("🛡️ Behavioral Anomaly Detection — Analyst Console")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Events processed", f"{len(all_events):,}")
col2.metric("Alerts (top 2% risk budget)", f"{len(alerts):,}")
col3.metric("Distinct entities flagged", f"{alerts['entity_id'].nunique():,}")
col4.metric("Avg. risk score of alerts", f"{alerts['risk_score'].mean():.1f}")

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("Ranked Alert Queue")
    type_filter = st.multiselect("Filter by predicted anomaly type",
                                  options=sorted(alerts["predicted_type"].unique()),
                                  default=sorted(alerts["predicted_type"].unique()))
    view = alerts[alerts["predicted_type"].isin(type_filter)].sort_values("risk_score", ascending=False)
    st.dataframe(
        view[["entity_id", "entity_type", "timestamp", "predicted_type",
              "type_confidence", "risk_score", "explanation"]],
        use_container_width=True, height=420,
    )

with right:
    st.subheader("Anomaly Type Breakdown")
    fig = px.pie(alerts, names="predicted_type", title="Alerts by predicted type")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("🔎 Entity Drill-down")
selected_entity = st.selectbox("Select an entity to inspect its history", sorted(all_events["entity_id"].unique()))
entity_hist = all_events[all_events["entity_id"] == selected_entity].sort_values("timestamp")

c1, c2 = st.columns([1, 1])
with c1:
    fig2 = px.line(entity_hist, x="timestamp", y="risk_score",
                    title=f"Risk score over time — {selected_entity}", markers=True)
    st.plotly_chart(fig2, use_container_width=True)
with c2:
    st.write("**Recent raw events**")
    st.dataframe(entity_hist[["timestamp", "resource_accessed", "auth_method",
                               "session_duration", "device_fingerprint", "risk_score"]].tail(15),
                 use_container_width=True)

matched_alert = alerts[alerts["entity_id"] == selected_entity]
if not matched_alert.empty:
    st.info("**Why this entity is flagged:** " + matched_alert.iloc[0]["explanation"])
else:
    st.success("No active high-risk alert for this entity.")
