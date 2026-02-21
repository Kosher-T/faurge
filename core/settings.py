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