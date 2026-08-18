# ══════════════════════════════════════════════════════════════════════════════
# Step 1ba — EQ Causality: Setup (Gain-Only)
# ══════════════════════════════════════════════════════════════════════════════
# Fixed EQ band (1kHz peak), predict only gain_db (1D output).
# Same as Step 1 but for EQ instead of volume.

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# ── What are we doing? ───────────────────────────────────────────────────────
N_CLIPS = 1
CLIP_PREFIXES = ["m", "f"]
DEGRADATIONS_PER_CLIP = 500

# Load Step 1b model (gain prediction) — replace head for EQ
STEP1B_MODEL = "/kaggle/input/notebooks/itorousa/ur-step-1/step_1/checkpoints/best_model.pt"

# Plugin paths (Kaggle cloud)
PLUGIN_DIR = Path('/kaggle/usr/lib/notebooks/itorousa')

# ── Fixed EQ band ────────────────────────────────────────────────────────────
EQ_FIXED_FREQ_HZ = 1000.0   # Fixed center frequency
EQ_FIXED_Q = 1.0            # Fixed Q
EQ_GAIN_RANGE_DB = (-12.0, 12.0)  # Only predict gain
N_EQ_BANDS = 1

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
OUTPUT_DIM = 1  # Just gain_db

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 300
PATIENCE = 100
TRAIN_SPLIT = 0.8
HIDDEN_DIM = 128

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT = Path('/kaggle/input')
PRISTINE = INPUT / 'notebooks/itorousa/daps-pristine'
OUTPUT = Path('/kaggle/working')
STEP1BA_DIR = OUTPUT / 'step_1ba'
CHECKPOINT_DIR = STEP1BA_DIR / 'checkpoints'
MODEL_PATH = CHECKPOINT_DIR / 'best_model.pt'
STEP1BA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU — using CPU (training will be slow)")
print(f"Device: {device}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nStep 1ba: EQ Causality (Gain-Only)")
print(f"Fixed EQ: {EQ_FIXED_FREQ_HZ}Hz, Q={EQ_FIXED_Q}")
print(f"Predict: gain_db only ({OUTPUT_DIM}D output)")
print(f"Gain range: {EQ_GAIN_RANGE_DB} dB")
print(f"Clips: {N_CLIPS} (prefixes={CLIP_PREFIXES})")
print(f"Step 1b model: {STEP1B_MODEL}")
