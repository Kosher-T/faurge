# ══════════════════════════════════════════════════════════════════════════════
# Step 1bb — Setup
# ══════════════════════════════════════════════════════════════════════════════
# Input:  135D (degraded metrics 67D + clean metrics 67D + frequency 1D)
# Output: 1D (gain_db)

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# ── What are we doing? ───────────────────────────────────────────────────────
N_CLIPS = 2
CLIP_PREFIXES = ["m", "f"]
DEGRADATIONS_PER_CLIP = 300

# Load Step 1ba champion (knows gain, doesn't know frequency)
STEP1BA_MODEL = "/kaggle/input/notebooks/itorousa/ur-step-1ba/step_1ba/checkpoints/curriculum_best.pt"

# Cache path — set to None to regenerate, or a path to load from
CACHE_PATH = None  # e.g. "/kaggle/input/my-cached-data/data.npz"

# Plugin paths (Kaggle cloud)
PLUGIN_DIR = Path('/kaggle/usr/lib/notebooks/itorousa')

# ── Fixed EQ band ────────────────────────────────────────────────────────────
EQ_FIXED_Q = 1.0
EQ_GAIN_RANGE_DB = (-12.0, 12.0)
N_EQ_BANDS = 1

# ── Frequency curriculum (50 frequencies) ────────────────────────────────────
FREQ_CURRICULUM = [
    # Octave bands
    100, 125, 160, 200, 250, 315, 400, 500, 630, 800,
    1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
    10000, 12500, 16000,
    # Fill gaps (between octave bands)
    110, 140, 180, 220, 280, 360, 450, 560, 710, 900,
    1100, 1400, 1800, 2200, 2800, 3600, 4500, 5600, 7100, 9000,
    11000, 14000, 18000,
    # Additional coverage
    150, 330, 470, 680, 3300, 4700, 6800,
]
FREQ_MIN_HZ = 20.0
FREQ_MAX_HZ = 20000.0

# ── Audio ─────────────────────────────────────────────────────────────────────
SR = 48000
CLIP_SEC = 5.0
CLIP_SAMPLES = int(SR * CLIP_SEC)

# ── Metrics ───────────────────────────────────────────────────────────────────
METRIC_DIM = 67
LTAS_DIM = 64
BARK_N_BANDS = 64
BARK_LOW_HZ = 20.0
BARK_HIGH_HZ = 20000.0
FFT_SIZE = 4096
HOP_SIZE = 1024

# ── Output dim ────────────────────────────────────────────────────────────────
OUTPUT_DIM = 1

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
LR = 5e-4
FREQ_EPOCHS = 100
FREQ_PATIENCE = 30
TRAIN_SPLIT = 0.8
HIDDEN_DIM = 128

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT = Path('/kaggle/input')
PRISTINE = INPUT / 'notebooks/itorousa/daps-pristine'
OUTPUT = Path('/kaggle/working')
STEP1BB_DIR = OUTPUT / 'step_1bb'
CHECKPOINT_DIR = STEP1BB_DIR / 'checkpoints'
DATA_DIR = STEP1BB_DIR / 'data'
MODEL_PATH = CHECKPOINT_DIR / 'curriculum_best.pt'
STEP1BB_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU — using CPU (training will be slow)")
print(f"Device: {device}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nStep 1bb: Frequency-Aware EQ Gain")
print(f"Input: 135D (134 metrics + 1 frequency)")
print(f"Output: 1D (gain_db)")
print(f"Load Step 1ba: {STEP1BA_MODEL}")
print(f"Clips: {N_CLIPS} (prefixes={CLIP_PREFIXES})")
print(f"Degradations per clip per freq: {DEGRADATIONS_PER_CLIP}")
print(f"Frequency curriculum: {len(FREQ_CURRICULUM)} frequencies")
print(f"Cache: {CACHE_PATH}")
