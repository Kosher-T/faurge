# ══════════════════════════════════════════════════════════════════════════════
# Step 0 — Metric Literacy: Setup
# ══════════════════════════════════════════════════════════════════════════════
# Teach Ursula to read metrics. No plugins. Just metric-to-metric.
# Input:  degraded metrics (67D)
# Output: clean metrics (67D)
#
# Step 0:  One clip, fresh training
# Step 0b: Multiple clips, load pretrained model from Step 0

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# ── What are we doing? ───────────────────────────────────────────────────────
N_CLIPS = 1                 # Step 0: 1 clip. Step 0b: 5 clips.
DEGRADATIONS_PER_CLIP = 200 # How many degraded versions per clean clip
LOAD_MODEL_FROM = None      # Step 0b: set to checkpoint path to fine-tune

# ── Plugin ranges (linear only — nonlinear belongs to Genesis) ───────────────
# Gain
GAIN_RANGE_DB = (-12.0, 12.0)

# EQ (parametric, random bands)
EQ_N_BANDS_RANGE = (1, 6)      # How many random EQ bands per degradation
EQ_GAIN_RANGE_DB = (-6.0, 6.0) # Gain per band
EQ_FREQ_RANGE_HZ = (80.0, 8000.0)  # Center frequency range
EQ_Q_RANGE = (0.5, 3.0)        # Quality factor (bandwidth)

# ── Audio ─────────────────────────────────────────────────────────────────────
SR = 48000
CLIP_SEC = 5.0
CLIP_SAMPLES = int(SR * CLIP_SEC)  # 240,000

# ── Metrics ───────────────────────────────────────────────────────────────────
METRIC_DIM = 67       # LTAS(64) + LUFS(1) + Crest(1) + ZCR(1)
LTAS_DIM = 64
BARK_N_BANDS = 64
BARK_LOW_HZ = 20.0
BARK_HIGH_HZ = 20000.0
FFT_SIZE = 4096
HOP_SIZE = 1024

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 30
PATIENCE = 8
TRAIN_SPLIT = 0.8
HIDDEN_DIM = 128

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT = Path('/kaggle/input')
PRISTINE = INPUT / 'notebooks/itorousa/daps-pristine'
OUTPUT = Path('/kaggle/working')
STEP0_DIR = OUTPUT / 'step_0'
CHECKPOINT_DIR = STEP0_DIR / 'checkpoints'
MODEL_PATH = CHECKPOINT_DIR / 'best_model.pt'  # Where to save/load model
STEP0_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU — using CPU (training will be slow)")
print(f"Device: {device}")
