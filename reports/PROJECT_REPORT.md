# Project Report — BehaviorGuard: AI-Powered Behavioral Anomaly Detection

## 1. Assumptions
1. Real intrusion/access-log datasets are unavailable for this exercise (scarce,
   privacy-restricted, or domain-specific per the problem statement), so all
   development and evaluation uses a synthetic dataset generated with a documented
   schema and injected, ground-truth-labeled attack taxonomy.
2. The same behavioral schema (entity_id, timestamp, source_ip, geo_location,
   resource_accessed, auth_method, session_duration, command_sequence,
   device_fingerprint) can represent a cloud user, a service account, or an
   edge/IoT device — the detection task is treated as domain-agnostic.
3. Ground-truth labels are used only for training/evaluation and are hidden from
   the models at inference time (`*_unlabeled.csv` output of the generator).
4. "Cold start" is defined as an entity with fewer than 15 observed events.

## 2. Data
`generate_synthetic_data.py` builds a stable per-entity habitual profile (typical
login hours, home geo-location, typical resources, device fingerprint), then
samples ~3 normal events/entity/day with realistic noise, and injects 7 attack
patterns at a controlled rate (brute force, impossible travel, credential
stuffing, lateral movement, device spoofing, low-and-slow exfiltration, and
insider drift as an ambiguous false-positive-tuning edge case).

Example run (150 entities, 15 days): 7,285 total events, ~6.4% anomalous
(injected attacks are deliberately over-sampled relative to the real-world
"tiny fraction" to keep classifier training tractable on small synthetic runs —
production training would use the full 0.5–3% injection rate at much larger
scale, and the risk-engine's percentile-based alert budget already handles the
imbalance rank-wise rather than by absolute count).

## 3. Modeling approach
| Component | Method | Why |
|---|---|---|
| Baseline Profiler | Per-entity statistical z-scores + population autoencoder, confidence-blended by history length | Solves cold start; statistical part is instantly explainable |
| Sequence Detector | LSTM autoencoder over rolling event windows | Captures order/timing anomalies (lateral movement, low-and-slow) that point-wise models miss |
| Anomaly Classifier | Gradient-boosted trees (`HistGradientBoostingClassifier`) with balanced class weights | Sample-efficient under class imbalance; produces clean feature importances for SHAP |
| Risk Engine | Weighted percentile blend of the three signals into a 0-100 score | Single, comparable score for the analyst regardless of which sub-model fired |
| Explainability | SHAP `TreeExplainer` on the classifier, top-k features rendered as plain language | Meets the "why was this flagged" requirement directly |

## 4. Evaluation approach (mapped to stated evaluation criteria)
- **Detection accuracy on imbalanced labels:** evaluated via per-class precision/
  recall/F1 (not raw accuracy, which is meaningless under imbalance) — see sample
  classification report in `run_pipeline.py` output.
- **Anomaly-type classification:** multi-class F1 per attack category.
- **False positive rate at realistic alert budget:** `risk_engine.alert_queue()`
  selects only the top 1–2% highest-risk events, mirroring a real SOC analyst's
  daily capacity, and FP rate should be measured against that fixed-size queue
  rather than the full event stream.
- **Explainability/usability:** every alert in the queue carries a generated
  reason string; the dashboard groups/filters by predicted type.
- **Cold start & concept drift:** handled architecturally (see Design Notes in
  README.md), not just evaluated post-hoc.
- **System design & scalability:** feature engineering is expressed as
  per-entity rolling computations, which map directly onto a streaming
  windowed-aggregation job (e.g., Flink/Kafka Streams) for real-time production
  use; the point-wise and sequence models can both score a new event in
  milliseconds once the rolling state is available.

## 5. Known limitations
- Synthetic attack patterns are hand-crafted approximations of real adversary
  behavior; an adaptive attacker could evade detection by mimicking normal
  statistics slowly (this is explicitly called out as "insider drift," an
  intentionally ambiguous edge case in the taxonomy).
- The LSTM sequence model and autoencoder are trained on the full (mostly
  normal) event stream rather than a curated clean-only set, since production
  systems rarely have a guaranteed-clean training window; this is standard
  practice for anomaly autoencoders (rare anomalies don't dominate the
  reconstruction objective) but is a simplification worth stress-testing on
  larger data.
- Current implementation profiles feature computation in batch (pandas); a
  production system would need incremental/streaming feature computation to
  hit true real-time latency at scale.
- Graph-based entity-resource relationship modeling (suggested in the problem
  statement as an alternative to LSTM/Transformer) is noted as future work but
  not implemented in this reference build.
