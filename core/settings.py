"""
core/settings.py
Global configuration for the Faurge pipeline.
Acts as the single source of truth for all paths, hardware limits, and audio standards.
Validates the loaded configuration against settings_schema_v1.json at startup.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
import jsonschema

from core import defaults

# --- Setup & Environment Loading ---
# Load variables from .env if present. Existing env vars take precedence.
# This must happen before we define our constants.
load_dotenv()

# --- Directory Structure ---
# Anchors the paths to the root faurge/ directory
BASE_DIR = Path(__file__).resolve().parent.parent

DIRS = {
    "SHADOW": BASE_DIR / "shadow",
    "MODELS": BASE_DIR / "models",
    "DSP_IR": BASE_DIR / "dsp" / "ir_library",
    "DSP_PRESETS": BASE_DIR / "dsp" / "presets",
    "DATA": BASE_DIR / "data",
    "STATE": BASE_DIR / "state",
    # MLOps & System Directories
    "ROLLBACK": BASE_DIR / "dsp" / "rollback",
    "CHECKPOINTS": BASE_DIR / "checkpoints",
    "LOGS": BASE_DIR / "logs",
}

# Ensure core runtime directories exist
for path in DIRS.values():
    path.mkdir(parents=True, exist_ok=True)


# --- Core Configuration Loader ---
def _load_int(key: str, default: int) -> int:
    return int(os.getenv(key, default))

def _load_float(key: str, default: float) -> float:
    return float(os.getenv(key, default))

def _load_bool(key: str, default: bool) -> bool:
    val = str(os.getenv(key, default)).lower()
    return val in ('true', '1', 't', 'y', 'yes')

def _load_str(key: str, default: str) -> str:
    return os.getenv(key, default)


# --- Hardware & VRAM Watchdog Constraints ---
GPU_TOTAL_VRAM_MB = _load_int("FAURGE_GPU_TOTAL_VRAM_MB", defaults.GPU_TOTAL_VRAM_MB)
VRAM_SAFETY_MARGIN_MB = _load_int("FAURGE_VRAM_SAFETY_MARGIN_MB", defaults.VRAM_SAFETY_MARGIN_MB)
WATCHDOG_POLL_RATE_SEC = _load_float("FAURGE_WATCHDOG_POLL_RATE_SEC", defaults.WATCHDOG_POLL_RATE_SEC)

# --- Audio & Shadow Space Configuration ---
AUDIO_SAMPLE_RATE = _load_int("FAURGE_AUDIO_SAMPLE_RATE", defaults.AUDIO_SAMPLE_RATE)
AUDIO_CHANNELS = _load_int("FAURGE_AUDIO_CHANNELS", defaults.AUDIO_CHANNELS)
SHADOW_BUFFER_LENGTH_SEC = _load_float("FAURGE_SHADOW_BUFFER_LENGTH_SEC", defaults.SHADOW_BUFFER_LENGTH_SEC)

# --- DSP & PipeWire API Settings ---
DSP_OSC_IP = _load_str("FAURGE_DSP_OSC_IP", defaults.DSP_OSC_IP)
DSP_OSC_PORT = _load_int("FAURGE_DSP_OSC_PORT", defaults.DSP_OSC_PORT)
MAX_GAIN_DB = _load_float("FAURGE_MAX_GAIN_DB", defaults.MAX_GAIN_DB)
MIN_GAIN_DB = _load_float("FAURGE_MIN_GAIN_DB", defaults.MIN_GAIN_DB)
CLIP_THRESHOLD_DBFS = _load_float("FAURGE_CLIP_THRESHOLD_DBFS", defaults.CLIP_THRESHOLD_DBFS)

# --- Fallback / Baseline Metrics ---
FALLBACK_TARGETS = {
    "LUFS": _load_float("FAURGE_FALLBACK_TARGETS_LUFS", defaults.FALLBACK_TARGETS_LUFS),
    "LRA": _load_float("FAURGE_FALLBACK_TARGETS_LRA", defaults.FALLBACK_TARGETS_LRA),
    "CREST_FACTOR": _load_float("FAURGE_FALLBACK_TARGETS_CREST_FACTOR", defaults.FALLBACK_TARGETS_CREST_FACTOR),
}

# --- Neural Network Training Hyperparameters ---
# Currently fixed array per Phase 1 roadmap, but agent parameters are adjustable
FFT_SIZES = [2048, 1024, 512, 256, 128, 64]
HARMONIC_OSCILLATORS = _load_int("FAURGE_HARMONIC_OSCILLATORS", defaults.HARMONIC_OSCILLATORS)

# --- Live Loop & MLOps ---
BYPASS_ON_VRAM_SPIKE = _load_bool("FAURGE_BYPASS_ON_VRAM_SPIKE", defaults.BYPASS_ON_VRAM_SPIKE)
ROLLBACK_RETENTION = _load_int("FAURGE_ROLLBACK_RETENTION", defaults.ROLLBACK_RETENTION)
MAX_LATENCY_MS = _load_int("FAURGE_MAX_LATENCY_MS", defaults.MAX_LATENCY_MS)

# --- API & Telemetry ---
HEALTH_HTTP_PORT = _load_int("FAURGE_HEALTH_PORT", defaults.HEALTH_HTTP_PORT)
METRICS_PORT = _load_int("FAURGE_METRICS_PORT", defaults.METRICS_PORT)
API_KEY = _load_str("FAURGE_API_KEY", defaults.API_KEY)

# --- Logging ---
LOG_LEVEL = _load_str("FAURGE_LOG_LEVEL", defaults.LOG_LEVEL)
LOG_MAX_BYTES = _load_int("FAURGE_LOG_MAX_BYTES", defaults.LOG_MAX_BYTES)
LOG_BACKUP_COUNT = _load_int("FAURGE_LOG_BACKUP_COUNT", defaults.LOG_BACKUP_COUNT)

# --- Phase 2: Memory & Budget Enforcement ---
VRAM_BUDGET_LIMIT_MB = _load_int("FAURGE_VRAM_BUDGET_LIMIT_MB", defaults.VRAM_BUDGET_LIMIT_MB)
RAM_BUDGET_LIMIT_MB = _load_int("FAURGE_RAM_BUDGET_LIMIT_MB", defaults.RAM_BUDGET_LIMIT_MB)
BUDGET_INFERENCE_STEPS = _load_int("FAURGE_BUDGET_INFERENCE_STEPS", defaults.BUDGET_INFERENCE_STEPS)
BUDGET_WARMUP_STEPS = _load_int("FAURGE_BUDGET_WARMUP_STEPS", defaults.BUDGET_WARMUP_STEPS)
BUDGET_ABORT_ON_EXCEED = _load_bool("FAURGE_BUDGET_ABORT_ON_EXCEED", defaults.BUDGET_ABORT_ON_EXCEED)


# ==============================================================================
# --- Configuration Validation ---
# ==============================================================================

def validate_settings():
    """Validates the current settings against the JSON Schema."""
    schema_path = BASE_DIR / "core" / "schemas" / "settings_schema_v1.json"
    if not schema_path.exists():
        # Fallback for tests or missing schema files
        return

    with open(schema_path, "r") as f:
        schema = json.load(f)

    # Build a snapshot of the current flat variables for validation
    config_snapshot = {
        "GPU_TOTAL_VRAM_MB": GPU_TOTAL_VRAM_MB,
        "VRAM_SAFETY_MARGIN_MB": VRAM_SAFETY_MARGIN_MB,
        "WATCHDOG_POLL_RATE_SEC": WATCHDOG_POLL_RATE_SEC,
        "AUDIO_SAMPLE_RATE": AUDIO_SAMPLE_RATE,
        "AUDIO_CHANNELS": AUDIO_CHANNELS,
        "SHADOW_BUFFER_LENGTH_SEC": SHADOW_BUFFER_LENGTH_SEC,
        "DSP_OSC_IP": DSP_OSC_IP,
        "DSP_OSC_PORT": DSP_OSC_PORT,
        "MAX_GAIN_DB": MAX_GAIN_DB,
        "MIN_GAIN_DB": MIN_GAIN_DB,
        "CLIP_THRESHOLD_DBFS": CLIP_THRESHOLD_DBFS,
        "FALLBACK_TARGETS": FALLBACK_TARGETS,
        "FFT_SIZES": FFT_SIZES,
        "HARMONIC_OSCILLATORS": HARMONIC_OSCILLATORS,
        "BYPASS_ON_VRAM_SPIKE": BYPASS_ON_VRAM_SPIKE,
        "ROLLBACK_RETENTION": ROLLBACK_RETENTION,
        "MAX_LATENCY_MS": MAX_LATENCY_MS,
        "HEALTH_HTTP_PORT": HEALTH_HTTP_PORT,
        "METRICS_PORT": METRICS_PORT,
        "LOG_LEVEL": LOG_LEVEL,
        "LOG_MAX_BYTES": LOG_MAX_BYTES,
        "LOG_BACKUP_COUNT": LOG_BACKUP_COUNT,
        "VRAM_BUDGET_LIMIT_MB": VRAM_BUDGET_LIMIT_MB,
        "RAM_BUDGET_LIMIT_MB": RAM_BUDGET_LIMIT_MB,
        "BUDGET_INFERENCE_STEPS": BUDGET_INFERENCE_STEPS,
        "BUDGET_WARMUP_STEPS": BUDGET_WARMUP_STEPS,
        "BUDGET_ABORT_ON_EXCEED": BUDGET_ABORT_ON_EXCEED,
    }

    try:
        jsonschema.validate(instance=config_snapshot, schema=schema)
    except jsonschema.exceptions.ValidationError as e:
        raise RuntimeError(f"Faurge Configuration Error: {e.message}") from e

# Validate on import
validate_settings()