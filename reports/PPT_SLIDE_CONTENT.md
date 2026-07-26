# Slide Content — AI-Powered Behavioral Anomaly Detection for Cybersecurity
(Fill directly into `IDEA_Presentation_Format.pptx`. Delete Slide 1 — instructions — before submission.)

---

## Slide 2 — TITLE PAGE
- **Problem Statement ID:** *(fill in from portal)*
- **Problem Statement Title:** AI-Powered Behavioral Anomaly Detection for Cybersecurity
- **Theme:** Cybersecurity / Smart Automation
- **PS Category:** Software
- **Team Name / Student Name / ID:** *(fill in your details)*

---

## Slide 3 — IDEA TITLE
**Title:** BehaviorGuard — Domain-Agnostic Behavioral Anomaly Detection Engine

**Proposed Solution:**
- A three-stage ML pipeline that learns *per-entity* normal behaviour (user, service account, or edge/IoT device) directly from access/connection logs and flags deviations in near real-time — the same engine works unmodified across IT, OT, and edge environments because it operates on a domain-agnostic event schema, not raw packets.
- **Stage 1 — Baseline Profiler:** blends a lightweight per-entity statistical profile (works from event #1) with a population-level autoencoder that learns the manifold of "normal" once enough history exists — solving cold-start cleanly via a confidence-weighted blend.
- **Stage 2 — Sequence Detector:** an LSTM autoencoder reads a rolling window of an entity's last N events and flags deviations in *order and timing*, not just single-event outliers — catching lateral movement and low-and-slow exfiltration that point-wise models miss.
- **Stage 3 — Risk Engine + Classifier:** a gradient-boosted classifier labels the anomaly type (brute force, impossible travel, credential stuffing, lateral movement, device spoofing, exfiltration), and a weighted risk engine combines all signals into one 0–100 analyst-facing score.

**How it addresses the problem:**
- Class imbalance → trained with class-balanced weighting + evaluated at a realistic analyst alert budget (top 1–2% of events), not raw accuracy.
- Concept drift → rolling per-entity windows mean "normal" continuously re-baselines as legitimate behaviour evolves.
- Explainability → every alert carries a SHAP-based, human-readable reason string (e.g., "flagged due to geo-velocity spike + new device fingerprint").
- Cold-start → new entities are scored against their *entity-type* population prior until they accumulate enough history for a personalized profile.

**Innovation / uniqueness:**
- Single unified engine across IT servers, OT/industrial edge gateways, POS terminals, and home IoT hubs — the ML task (sequence anomaly detection on access events) is identical regardless of the underlying infrastructure.
- Cold-start blending formula (statistical prior ↔ learned autoencoder profile) is a lightweight, explainable alternative to more opaque few-shot approaches.

---

## Slide 4 — TECHNICAL APPROACH
**Technologies:** Python, PyTorch (autoencoder + LSTM sequence model), scikit-learn `HistGradientBoostingClassifier` (anomaly-type classification), SHAP (explainability), pandas/NumPy + Faker (synthetic data generation), Streamlit + Plotly (analyst dashboard), Kafka/Kinesis (conceptual streaming ingestion for production).

**Methodology / pipeline (see architecture diagram):**
1. Synthetic access-log generator (per-entity habitual baseline + 7 injected attack patterns at 0.5–3% rate, ground truth retained separately).
2. Per-entity feature engineering: rolling z-scored access hour, geo-velocity (haversine distance / time), IP/fingerprint change flags, resource novelty, auth-failure rate, event burst rate.
3. Parallel scoring: Baseline Profiler (point-wise) + Sequence Detector (LSTM, window-wise) run concurrently.
4. Anomaly-type Classifier assigns a category + confidence to flagged events.
5. Risk Engine blends all three into one ranked, percentile-based risk score.
6. Explainability layer (SHAP) attaches a plain-language reason to every alert.
7. Analyst dashboard surfaces the ranked queue, type breakdown, and per-entity drill-down history.

*(Insert `architecture_diagram.png` here — full pipeline flow.)*

---

## Slide 5 — FEASIBILITY AND VIABILITY
**Feasibility:**
- All components use mature, well-supported open-source libraries (PyTorch, scikit-learn, SHAP) — no exotic infrastructure required.
- The feature set (rolling per-entity statistics) is computable incrementally in a streaming setting (Kafka/Flink), so the design maps directly onto a real-time SOC pipeline.
- Synthetic-data-first approach sidesteps the scarcity/privacy problem of real intrusion datasets while still validating the full pipeline end-to-end.

**Potential challenges & risks:**
- *Synthetic-to-real gap:* attack patterns simulated in synthetic data may not capture the full diversity of real adversary behaviour.
- *Alert fatigue:* even at low false-positive rates, high event volume can overwhelm analysts.
- *Adaptive/evasive attackers:* slow, mimicry-style attacks could be engineered to stay within learned "normal" bounds.
- *Cold-start ambiguity:* brand-new entities have inherently weak signal regardless of scoring strategy.

**Mitigation strategies:**
- Continuously retrain/re-baseline on production data (with human-in-the-loop labeling from SOC feedback) once deployed, closing the synthetic-to-real gap over time.
- Enforce a fixed analyst alert budget (e.g., top 1% of events) rather than a fixed threshold, keeping alert volume manageable regardless of traffic growth.
- Combine sequence modeling with graph-based entity-resource relationship modeling (future work) to catch slow, distributed attacks that a single-entity view misses.
- Use the entity-type population prior (not zero-knowledge) for cold-start entities, and flag their scores as "low-confidence" to the analyst rather than hiding or over-trusting them.

---

## Slide 6 — ARTIFACTS
- **Code:** synthetic data generator, feature engineering, baseline profiler (autoencoder), sequence detector (LSTM autoencoder), anomaly classifier (gradient boosted trees), risk engine, SHAP explainability layer, Streamlit dashboard — all included in the submitted code archive.
- **Dashboard snapshot:** *(insert `dashboard_mockup.png`, or a live screenshot from `streamlit run dashboard/app.py`)* — ranked alert queue, anomaly-type breakdown, and entity risk-history drill-down.
- **Sample output:** ranked, explained alerts in `reports/scored_alerts.csv`, e.g. *"Flagged due to: unusual gap since last activity (+7.40); access to a never-before-seen resource (+4.22)."*

---

## Slide 7 — RESEARCH AND REFERENCES
- Malhotra, P. et al., *"Long Short Term Memory Networks for Anomaly Detection in Time Series"*, ESANN 2015.
- Chandola, V., Banerjee, A., Kumar, V., *"Anomaly Detection: A Survey"*, ACM Computing Surveys, 2009.
- Lundberg, S. & Lee, S.-I., *"A Unified Approach to Interpreting Model Predictions"* (SHAP), NeurIPS 2017.
- MITRE ATT&CK Framework — techniques referenced: Credential Access (T1110 Brute Force), Lateral Movement (TA0008), Exfiltration (TA0010). https://attack.mitre.org
- OWASP guidance on credential stuffing and impossible-travel detection patterns.
- scikit-learn, PyTorch, and SHAP official documentation (implementation reference).

