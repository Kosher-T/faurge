Files Found

- core/bake_orchestrator.py — Fabian orchestrator: metric extraction, `bake()` calls `run_ursula()` with assembled metrics.
- core/clap_utils.py — CLAP helpers and cosine similarity utilities.
- core/schemas/settings_schema_v1.json — settings schema; place to add optional config keys.
- docs/ursula.md — Ursula input/observation description: specifies (M_degraded, M_ref) = 67D concatenation.
- README.md / docs/dsp.md — architecture notes and 67D definition.
- kaggle/02/_build_01_notebook.py — batch serialization: saves arrays and `batch_meta` JSON (training data path).
- plugins_port/gain/gain.py — example plugin that returns metadata dicts (shows how metric metadata is packaged).

Key Symbols

- `BakeOrchestrator.extract_physical_metrics(self, audio, sr=48000) -> Dict[str, Any]`
  - Returns `'ltas'`, `'lufs'`, `'dyn_range'` (basis of the 67D vector).
- `BakeOrchestrator.run_ursula(self, input_metrics: Dict, reference_metrics: Dict) -> Dict[str,float]`
  - Receives metric dicts (where new `cluster` key would be consumed).
- `core/clap_utils.encode_audio(...) -> np.ndarray` and `cosine_similarity(a,b) -> float` (used for similarity checks).
- Kaggle batch writer: `np.savez(...)` and `json.dump(batch_meta)` — spot where extra metadata (cluster labels) can be persisted.

Where 67D is built / passed

- README/docs: "Input = M_current || M_ref (M ∈ R^{67})" — source = LTAS(64) + LUFS(1) + DynRange(2).
- `extract_physical_metrics` returns separate components; later concatenation happens in `bake()` before `run_ursula()`.
- Training pipeline saves `target_clap`/`batch_meta` — training data ingestion is where extra labels can be added.

Schemas

- `core/schemas/settings_schema_v1.json` currently lacks cluster-related keys. Add optional fields like `CLUSTER_DEFAULT` and `ENABLE_CLUSTER_PAIR` to enable runtime toggles and defaults without breaking validation.

Tests & Implications

- `tests/test_core_config.py` validates settings schema: schema additions require updating tests or marking new fields optional.
- `plugins_port/*` tests (e.g., `test_gain.py`) assert presence of certain metadata keys; adding optional `cluster` keys in metric dicts should not break them if kept optional.
- Training/ingestion code not covered by unit tests here will need updates if batch metadata format changes.

Concrete Insertion Points (pros/cons)

1) Add `cluster` to `extract_physical_metrics` (recommended)
   - Pros: Centralized; Ursula receives richer inputs natively; minimal changes to call sites if consumers accept optional keys.
   - Cons: Requires adding cluster-detection logic (or passing through from metadata), and updating any strict consumers that assume exact keys or vector shapes.

2) Add `cluster` only to Kaggle batch metadata (training-only)
   - Pros: Least invasive to runtime; gives training data explicit labels so models can learn cluster constraints.
   - Cons: Runtime Ursula won't see cluster data unless ingestion/serving code is updated; only helps future re-training and supervised learning.

3) Add optional config/schema entries first
   - Pros: Safe, backwards-compatible approach enabling toggles and defaults; enforces validation and documents behavior.
   - Cons: Does not by itself inject cluster into inputs; follow-up code changes needed.

Recommendations & Next Steps

Short-term (minimal disruption):
- Implement optional `cluster` key in `extract_physical_metrics` return dict. Keep it optional; do not change the 67D concatenation shape for existing tests — instead attach `cluster` as a separate metadata key alongside the 67D vector.
- Update `core/schemas/settings_schema_v1.json` with optional cluster-related config entries (e.g., `CLUSTER_DEFAULT`, `CLUSTER_PROVIDER`) and adjust `core/settings.py` to surface defaults.
- Add `cluster` to `batch_meta` in the Kaggle data pipeline so historical training data includes cluster labels.
- Update unit tests that validate settings schema and any strict metric-key tests to tolerate the optional `cluster` key.

Longer-term:
- Add a small `cluster` detection module (heuristic or classifier) that assigns cluster IDs (e.g., `male`, `female`, `speech`, `music`, or numeric cluster IDs) and document cluster ontology in `docs/ursula.md`.
- Retrain/update Ursula models to condition on the `cluster_pair` or to learn per-cluster similarity thresholds.

Action choices (pick one for me to implement next):
A) Draft the exact JSON schema snippet and `core/settings.py` change required to add optional cluster config entries.
B) Draft and implement a prototype `extract_physical_metrics` change returning `{ '67d': np.ndarray, 'cluster': Optional[str], ... }` and update `bake()` concatenation to pass the metadata to `run_ursula()`.
C) Patch the Kaggle batch writer to include `cluster` in `batch_meta` and update the Kaggle notebook's metadata schema.

I started a tracked TODO list with these steps. Which action (A/B/C) should I do now?