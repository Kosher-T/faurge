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

### Cloud Dataset Preparation (Kaggle Environment)
*All files here live in the local `kaggle/` directory but are executed in the cloud.*
- ~~**`kaggle/01_acquire_and_augment.ipynb`**: Downloads VCTK/LJSpeech directly to Kaggle's high-speed storage. Convolves dry vocals with IRs and injects noise.~~
- **`kaggle/02_generate_physical_labels.ipynb`**: Iterates cloud audio files to calculate ground-truth LTAS, LUFS, LRA, and Crest Factor values for Fabian's training.
- **`kaggle/dataset_manifest.json`**: A lightweight manifest generated in the cloud, tracking checksums. Tracked locally by Git to monitor dataset versions without downloading the 40GB audio files.
- **`scripts/validate_manifest.py`**: Local script that asserts the Git-tracked manifest is well-formed, contains expected checksums, and the dataset version hasn't drifted. Strengthens the Phase 2 → 3 gate.

### Cloud Model Acquisition & Freezing (Kaggle Environment)
- **`kaggle/03_freeze_and_export_backbones.ipynb`**: Downloads the massive CLAP text/audio encoders to Kaggle. Strips out unnecessary training layers, freezes the weights, and packages them into a compressed `.tar.gz` archive for low-bandwidth local downloading.

### Local Artifact Ingestion
- **`scripts/ingest_cloud_artifacts.py`**: A local script designed to cleanly unpack the frozen models downloaded from Kaggle into the local `models/` directory, verifying their SHA256 hashes before allowing Phase 3 to run.

---

## Phase 3: The Cognitive Layer (Cloud Training & Offline Sandbox)

We build and train the tensor networks in the cloud. Checkpoints are optimized, quantized, and exported as immutable artifacts for the edge device.

### Agent Blueprints (Local & Cloud Shared)
- **`config/hyperparameters.yaml`**: The single source of truth for learning rates and network dimensions.
- **`agents/base_agent.py`, `fabian.py`, `ursula.py`, `genesis.py`**: The core AI logic. These files are pushed to Kaggle via GitHub for training, but execute locally during live inference.

### Ursula's Offline Sandbox
- **`core/gym_env.py`**: Ursula's DSP simulator. 
  - *Note for Kaggle:* Since Kaggle does not have PipeWire, this environment will use a Python-native DSP library (like `pedalboard` or headless `pysndfx`) to simulate EQ and Compression during cloud training.

### Cloud Training Orchestration (Kaggle Environment)
- **`kaggle/requirements-kaggle.txt`**: Cloud-specific dependencies (`pedalboard`, `pyloudnorm`, audio libs) `pip install`-ed at the top of each notebook.
- **`kaggle/04_train_fabian.ipynb`**: Trains the Routing and Physical Regression heads using the cloud-generated labels.
- **`kaggle/05_train_ursula.ipynb`**: Runs the Soft Actor-Critic RL loop inside the headless Python DSP simulator.
- **`kaggle/06_train_genesis.ipynb`**: Trains the DDSP synthesizer against real-world IRs.
- **`kaggle/07_quantize_and_export.ipynb`**: Runs post-training float16/INT8 quantization on the final models. Exports them as lightweight ONNX/TFLite weights, ready to be zipped and downloaded to the local edge machine.

---

## Phase 4: The Live Loop (Local Orchestration & Execution)

The final phase glues the trained networks (downloaded from Kaggle) to the local PipeWire infrastructure with extreme fault tolerance.

### Core Execution & Safety
- **`faurge.py` & `systemd/faurge-main.service`**: The root execution daemon. 
- **`core/bake_orchestrator.py`**: The sequential Fabian → Ursula → Genesis pipeline. Wakes agents, runs inference, writes IRs to disk, and guarantees full VRAM unload.
- **`core/audio_processor.py` & `core/state_manager.py`**: Encapsulates real-time processing, buffer math, and smooth cross-fade logic for hot-swapping baked IRs.
- **`core/signal_handlers.py` & `core/kill_switch.py`**: Ensures graceful teardowns and provides an instant hardware bypass.

### Monitoring & Rollout
- **`web_api/app.py`**: API-key secured server exposing status metrics and bypass controls.
- **`core/baker.py` & `scripts/backup_rollback_policy.sh`**: Writes IRs to disk and manages automated rollback snapshots.

### Local Validation Gates
- **`tests/test_live_loop.py`**: End-to-end simulations validating bypass fallback under massive GPU spikes and asserting <20ms latency under load.
- **`tests/test_vram_unload.py`**: Asserts GPU memory returns to baseline after a complete local bake cycle.

---

## Cross-Phase Project Governance

- **CI/CD (`.github/workflows/ci.yml`)**: Matrix testing (GPU + CPU-only), caching, and simulated VRAM-budget checks on every push.
- **CPU-Only Deployment**: All agents and the bake pipeline must function without a GPU. CI includes a dedicated CPU-only matrix leg to catch regressions.
- **Developer Experience**: Black formatting, Ruff linting, and Mypy type safety via pre-commit hooks.
- **VRAM Lifecycle**: `tests/test_vram_unload.py` asserts GPU memory returns to baseline after a complete bake cycle. Skipped in CPU-only mode.

---

## Mandatory Phase Gates

| Gate | Condition |
|------|-----------|
| Phase 1 → 2 | `test_pipewire_routing.py` cleanly exits; hardware dry bypass is confirmed. |
| Phase 2 → 3 | VRAM budget audited locally. `validate_manifest.py` passes. Frozen backbones securely downloaded to edge device. |
| Phase 3 → 4 | Kaggle training completes. Quantized models downloaded and pass local SHA256 verification. |
| Live Bake | Immutable bake manifest committed. `test_vram_unload.py` confirms clean teardown. |