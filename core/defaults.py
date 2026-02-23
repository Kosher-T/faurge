"""
core/defaults.py
The canonical registry of all Faurge default configuration values.
This file contains no logic, only constants.
"""

# --- Hardware & VRAM Watchdog Constraints ---
GPU_TOTAL_VRAM_MB = 4096
VRAM_SAFETY_MARGIN_MB = 500
WATCHDOG_POLL_RATE_SEC = 2.0

# --- Audio & Shadow Space Configuration ---
AUDIO_SAMPLE_RATE = 48000
AUDIO_CHANNELS = 1
SHADOW_BUFFER_LENGTH_SEC = 5.0

# --- DSP & PipeWire API Settings ---
DSP_OSC_IP = "127.0.0.1"
DSP_OSC_PORT = 9000
MAX_GAIN_DB = 15.0
MIN_GAIN_DB = -24.0
CLIP_THRESHOLD_DBFS = -0.5

# --- Fallback / Baseline Metrics ---
FALLBACK_TARGETS_LUFS = -16.0
FALLBACK_TARGETS_LRA = 4.0
FALLBACK_TARGETS_CREST_FACTOR = 3.5

# --- Neural Network Training Hyperparameters ---
HARMONIC_OSCILLATORS = 100

# --- Live Loop, Health, and MLOps Configuration ---
BYPASS_ON_VRAM_SPIKE = True
ROLLBACK_RETENTION = 10

# --- Latency & Real-Time Constraints ---
MAX_LATENCY_MS = 20

# --- API & Telemetry ---
HEALTH_HTTP_PORT = 8765
METRICS_PORT = 8000
API_KEY = ""

# --- Logging ---
LOG_LEVEL = "INFO"
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 5

# --- Phase 2: Memory & Budget Enforcement ---
VRAM_BUDGET_LIMIT_MB = 3900       # Hard peak-VRAM ceiling (≈ 3.9 GB)
RAM_BUDGET_LIMIT_MB = 4096        # CPU-only RAM ceiling
BUDGET_INFERENCE_STEPS = 5        # Inference steps during profiling
BUDGET_WARMUP_STEPS = 2           # Warmup steps before measurement
BUDGET_ABORT_ON_EXCEED = True     # Halt the build on budget overrun
