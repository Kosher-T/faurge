"""
core/settings.py
Global configuration for the Faurge pipeline.
Acts as the single source of truth for all paths, hardware limits, and audio standards.
"""

import os
from pathlib import Path

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
    # --- Added: MLOps & System Directories ---
    "ROLLBACK": BASE_DIR / "dsp" / "rollback",
    "CHECKPOINTS": BASE_DIR / "checkpoints",
    "LOGS": BASE_DIR / "logs",
}

# Ensure directories exist upon initialization
for path in DIRS.values():
    path.mkdir(parents=True, exist_ok=True)


# --- Hardware & VRAM Watchdog Constraints ---
# Configured for a strict 4GB VRAM limit
GPU_TOTAL_VRAM_MB = 4096  
VRAM_SAFETY_MARGIN_MB = 500  # Minimum free VRAM required before waking agents
WATCHDOG_POLL_RATE_SEC = 2.0 # How often the pynvml daemon checks the GPU


# --- Audio & Shadow Space Configuration ---
AUDIO_SAMPLE_RATE = 48000  # Standard for broadcast/video
AUDIO_CHANNELS = 1         # Processing mono vocal inputs
SHADOW_BUFFER_LENGTH_SEC = 5.0  # The length of the asynchronous capture


# --- DSP & PipeWire API Settings ---
# Defaulting to standard local OSC ports for headless LV2/Carla manipulation
DSP_OSC_IP = "127.0.0.1"
DSP_OSC_PORT = 9000

# Ursula's physical constraints to prevent catastrophic audio blowouts
MAX_GAIN_DB = 15.0
MIN_GAIN_DB = -24.0
CLIP_THRESHOLD_DBFS = -0.5


# --- Fallback / Baseline Metrics ---
# If Fabian's routing fails or is bypassed, Ursula defaults to these physical targets
FALLBACK_TARGETS = {
    "LUFS": -16.0,          # Standard podcast/broadcast loudness
    "LRA": 4.0,             # Tight dynamic range
    "CREST_FACTOR": 3.5,    # Smooth, non-harsh transients
}

# --- Neural Network Training Hyperparameters ---
# Genesis DDSP settings
FFT_SIZES = [2048, 1024, 512, 256, 128, 64] # For Multi-Scale Spectral Loss
HARMONIC_OSCILLATORS = 100


# ==============================================================================
# --- ADDED: Live Loop, Health, and MLOps Configuration ---
# ==============================================================================

# --- System Fallback & Rollback Policy ---
# Ensures live audio never drops frames if VRAM spikes or a bake degrades audio
BYPASS_ON_VRAM_SPIKE = os.getenv("FAURGE_BYPASS_ON_VRAM_SPIKE", "True").lower() == "true"
ROLLBACK_RETENTION = int(os.getenv("FAURGE_ROLLBACK_RETENTION", 10)) # Keep last 10 plugin snapshots

# --- Latency & Real-Time Constraints ---
# Watchdog will force dry-bypass if processing exceeds this round-trip time
MAX_LATENCY_MS = int(os.getenv("FAURGE_MAX_LATENCY_MS", 20))

# --- API & Telemetry ---
# Exposed ports for Prometheus scraping and GUI control endpoints
HEALTH_HTTP_PORT = int(os.getenv("FAURGE_HEALTH_PORT", 8765))
METRICS_PORT = int(os.getenv("FAURGE_METRICS_PORT", 8000))
API_KEY = os.getenv("FAURGE_API_KEY", "") # Leave empty for local dev without auth