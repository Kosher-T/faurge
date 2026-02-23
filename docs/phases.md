# Faurge Development Roadmap: The Four Phases

> **Development is currently for Linux OS only. Windows and MacOS versions coming soon.**

This document outlines the chronological execution plan for building Faurge. To prevent architectural collapse, ensure maintainability, and respect the strict 4GB VRAM constraints of the deployment hardware, development is divided into four strictly ordered phases. We build the physical plumbing, testing infrastructure, and observability pipelines before we attempt to turn on the mathematical water.

---

## Phase 1: The Shell (Infrastructure & Plumbing)

Before any AI wakes up, the system must establish bulletproof routing, hardware monitoring, and guaranteed dry-bypass fallbacks.

### Core Architecture & Config
- ~~**`requirements.txt` & `requirements-dev.txt`**: Strict separation of runtime dependencies and development tools.~~
- ~~**`core/settings.py` & `core/defaults.py`**: Global configuration loaded via env vars, enforcing absolute limits (e.g., `VRAM_LIMIT_GB=3.6`, `MAX_LATENCY_MS=20`).~~
- ~~**`core/schemas/settings_schema_v1.json`**: JSON Schema to validate settings, paired with `.env.example` for secure deployment.~~
- ~~**`core/logging.py`**: Structured, rotating logs for metrics and system states.~~

### Hardware & Audio Routing
- ~~**`core/watchdog.py` & `systemd/faurge-watchdog.service`**: The VRAM monitoring daemon running as a persistent, auto-restarting systemd service. Gracefully degrades on GPU-less systems.~~
- ~~**`core/hardware_detect.py`**: Probes for NVIDIA/AMD GPUs at startup. If none are found, forces Faurge into **CPU-only mode**, disabling VRAM monitoring and switching all tensor inference to CPU/OpenVINO.~~
- ~~**`core/shadow_capture.py`**: The asynchronous 5-second buffer recorder. Captures live audio into the Shadow Space for downstream agent analysis.~~
- ~~**`scripts/setup_pipewire.sh` & `scripts/reset_pipewire.sh`**: Establishes (and safely tears down) the virtual patch bay, hardwiring the live dry-bypass path.~~
- ~~**`scripts/setup_realtime_kernel.sh`**: Helper to configure low-latency kernels, `rtprio`, and PipeWire realtime permissions.~~

### Validation Gates & Health
- ~~**`scripts/validate_environment.py`**: Hard fails on missing dependencies, unsupported GPU drivers, or inadequate disk space.~~
- ~~**`scripts/pipewire_compat_check.py`**: Checks installed PipeWire/JACK versions against a known-stable compatibility matrix.~~
- ~~**`tests/test_pipewire_routing.py`**: Sends a known tone through the dry bypass to mathematically verify hardware-level passthrough.~~
- ~~**`core/health_endpoints.py`**: Exposes HTTP endpoints (`/health`, `/ready`, `/metrics`) to verify watchdog and audio server status.~~

---

## Phase 2: The Workbench (Data, Environment Prep & VRAM Accounting)

This phase enforces a strict VRAM budget audit before any model weights are written, and establishes the immutable datasets for training.

### Memory & Budget Enforcement
- ~~**`core/vram_policy.py` & `scripts/vram_budget.py`**: Loads models, runs a short inference loop to profile memory fragmentation, and halts the build if peak VRAM exceeds the safety margin (≥ 3.9GB). In CPU-only mode, the VRAM audit is skipped and RAM budget limits are enforced instead.~~
- ~~**`core/memory_tracker.py`**: Detailed peak/per-layer memory profiling to debug budget overruns. Supports both VRAM (pynvml) and system RAM tracking.~~
- ~~**`tests/test_vram_budget.py`**: Validates the budget logic using mock models, including the CPU-only fallback path.~~

### Dataset Preparation & Governance
- **`dvc.yaml` & `data/dataset_config.yaml`**: Uses DVC for dataset versioning, preventing hardcoded paths and storing checksums/provenance.
- **`scripts/download_datasets.py` & `scripts/prepare_dataset_ci.sh`**: Automates acquisition with resume support, and generates tiny subsets for CI testing.
- **`scripts/augment_dataset.py` & `scripts/generate_dataset.py`**: Convolves dry vocals with IRs, injecting noise and pitch shifts to harden Genesis's robustness.
- **`scripts/generate_physical_labels.py`**: Iterates audio files to calculate ground-truth LTAS, LUFS, LRA, and Crest Factor values. Prerequisite for Fabian's Physical Regression Head training.
- **`scripts/validate_dataset.py` & `tests/test_dataset_license.py`**: Post-generation checks ensuring all files are valid, physical labels are accurate, and legal licenses are compliant.

### Pretrained Model Acquisition
- **`scripts/download_pretrained.py`**: Downloads and hash-verifies frozen backbone weights (CLAP encoder, etc.) into `models/`. Must pass before Phase 3 training begins.

### Sandbox & Optimization
- **`core/gym_env.py` & `gym_env/test_env.py`**: Ursula's offline DSP simulator and its sanity checks. Depends on `scripts/setup_pipewire.sh` having validated a headless PipeWire/LV2 host.
- **`scripts/quantize_models.py`**: Post-training float16/INT8 quantization tools to drastically reduce VRAM footprints. Also exports CPU-optimized ONNX/OpenVINO variants.

---

## Phase 3: The Cognitive Layer (Agent Architecture & Training)

With data locked and VRAM secured, we build the tensor networks. All checkpoints are treated as immutable artifacts.

### Agent Blueprints
- **`config/hyperparameters.yaml`**: The single source of truth for all learning rates, network dimensions, and batch sizes.
- **`agents/base_agent.py`**: Abstract base class (load, save, predict, reset) to standardize agent API.
- **`agents/fabian.py`, `agents/ursula.py`, `agents/genesis.py`**: The core AI logic (Split-Router, SAC Tuner, DDSP Synthesizer).
- **`agents/compat_matrix.md`**: Documents agent behavior across float16/quantized/ONNX variants.

### Training & Orchestration
- **`training/train_all.py`**: Meta-script orchestrating the sequential training runs (Fabian → Ursula → Genesis).
- **`training/train_fabian.py`**: Freezes the CLAP text encoder and trains the Routing Head and Physical Regression Head MLPs on generated physical labels.
- **`training/tensorboard_logging.py` & `training/early_stopping.py`**: Universal loss curve tracking and overfitting prevention.
- **`mlflow/` (Model Registry)**: Documents promoted checkpoints, signature metadata, and deterministic commit SHAs.
- **`scripts/export_onnx.py` & `scripts/export_tflite.py`**: Inference optimization pipelines. Includes CPU-only export targets (ONNX Runtime / OpenVINO).

### Validation Gates & QA
- **`scripts/qa_metrics.py`**: Computes objective audio metrics (SI-SDR, STOI) to detect model regressions.
- **`tests/test_model_loading.py` & `tests/test_deterministic_repro.py`**: Verifies checkpoints load under budget and reproduce exact outputs given fixed RNG seeds.
- **`tests/test_cng_output.py`**: Validates Genesis's Comfort Noise Generator outputs (LFO rates, bandpass cutoffs, pink-noise amplitudes) have correct shapes and sane value ranges.

---

## Phase 4: The Live Loop (Orchestration & Execution)

The final phase glues the trained networks to the PipeWire infrastructure with extreme fault tolerance.

### Core Execution & Safety
- **`faurge.py` & `systemd/faurge-main.service`**: The root execution daemon. Detects CPU-only mode at startup and configures the pipeline accordingly.
- **`core/bake_orchestrator.py`**: The sequential Fabian → Ursula → Genesis pipeline. Wakes agents, runs inference, writes IRs/CNG seeds to disk, and guarantees full VRAM unload after each bake cycle.
- **`core/audio_processor.py` & `core/state_manager.py`**: Encapsulates real-time processing, buffer math, and the global bypass/active state machine. Implements smooth cross-fade logic to hot-swap baked IRs without audible pops.
- **`core/signal_handlers.py` & `core/kill_switch.py`**: Ensures graceful teardowns and provides a "panic" CLI/GPIO hook for instant hardware bypass.

### Monitoring, API, & Rollout
- **`web_api/app.py`**: API-key secured server exposing status metrics, bypass controls, and A/B testing over HTTP.
- **`monitoring/prometheus_exporter.py` & `monitoring/trace_alerts.md`**: Exports VRAM usage and bypass states, defining hard alert thresholds.
- **`scripts/canary_deploy.sh`**: Safely loads new checkpoints to a secondary process for A/B audio testing before full promotion.
- **`core/baker.py` & `scripts/backup_rollback_policy.sh`**: Writes IRs to disk, manages automated rollback snapshots, and enforces retention policies (e.g., keep last 10 bakes).

### Validation Gates
- **`tests/test_live_loop.py` & `tests/test_latency_under_load.py`**: End-to-end simulations validating bypass fallback under massive GPU spikes while maintaining <20ms latency.
- **`tests/regression/`**: Nightly CI suite comparing new model outputs against reference audio to catch drift.

---

## Cross-Phase Project Governance

- **CI/CD (`.github/workflows/ci.yml`)**: Matrix testing (GPU + CPU-only), caching, and simulated VRAM-budget checks on every push.
- **CPU-Only Deployment**: All agents and the bake pipeline must function without a GPU. CI includes a dedicated CPU-only matrix leg to catch regressions.
- **Developer Experience**: `devcontainer/` config, Black formatting, Ruff linting, and Mypy type safety via pre-commit hooks.
- **Observability**: OpenTelemetry tracing for critical asynchronous paths.
- **Legal & Privacy**: `docs/privacy.md` (telemetry notice, audio capture retention) and `legal/dataset_licenses.md`.
- **Release Management**: `RELEASE.md` outlining semantic versioning for both code and weights.
- **Auditability**: Every generated DSP bake produces an immutable manifest (SHA256 of model, timestamp, rollback pointer).
- **VRAM Lifecycle**: `tests/test_vram_unload.py` asserts GPU memory returns to baseline after a complete bake cycle. Skipped in CPU-only mode.

---

## Mandatory Phase Gates
No phase begins until the previous phase's gate condition is met.

| Gate | Condition |
|------|-----------|
| Phase 1 → 2 | `test_pipewire_routing.py` cleanly exits; hardware dry bypass is confirmed; `hardware_detect.py` sets GPU or CPU-only mode. |
| Phase 2 → 3 | `vram_budget.py` completes without exceeding the safety margin (or RAM budget in CPU-only mode). All datasets pass `validate_dataset.py`. Pretrained weights are hash-verified. |
| Phase 3 → 4 | Objective `qa_metrics.py` pass; checkpoints are registered in MLflow; `test_cng_output.py` passes. |
| Live Bake | Immutable bake manifest and rollback snapshot are committed. `test_vram_unload.py` confirms clean teardown. |