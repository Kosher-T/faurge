# ══════════════════════════════════════════════════════════════════════════════
# Step 2a — Tier 1 Metric Literacy (EQ Degradation): Setup
# ══════════════════════════════════════════════════════════════════════════════
# Teach the model Tier 1 spectral features by using EQ degradation.
# Gain-only doesn't change spectral shape — EQ does.
# Input:  degraded metrics (81D)
# Output: clean metrics (81D)
# Pretrained from step 1be (81D gain-only metric literacy).

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# ── What are we doing? ───────────────────────────────────────────────────────
N_CLIPS = 10
CLIPS_PER_GENDER = 5
DEGRADATIONS_PER_CLIP = 1000  # up from 500 — EQ has 2 DOF (freq + gain)

# Plugin paths (Kaggle cloud)
PLUGIN_DIR = Path('/kaggle/usr/lib/notebooks/itorousa')

# ── EQ settings ──────────────────────────────────────────────────────────────
EQ_FIXED_Q = 1.0
EQ_GAIN_RANGE_DB = (-12.0, 12.0)
EQ_FREQ_MIN_HZ = 20.0
EQ_FREQ_MAX_HZ = 18000.0

# ── Audio ─────────────────────────────────────────────────────────────────────
SR = 48000
CLIP_SEC = 5.0
CLIP_SAMPLES = int(SR * CLIP_SEC)

# ── Metrics ───────────────────────────────────────────────────────────────────
METRIC_DIM = 81  # 80D (Tier 0+1) + RMS Energy
LTAS_DIM = 64
BARK_LOW_HZ = 20.0
BARK_HIGH_HZ = 20000.0
FFT_SIZE = 4096
HOP_SIZE = 1024

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 200
PATIENCE = 30
TRAIN_SPLIT = 0.8
HIDDEN_DIM = 128

# ── Pretrained weights ───────────────────────────────────────────────────────
PRETRAINED_PATH = Path('/kaggle/input/notebooks/itorousa/ur-step-2/step_1be/checkpoints/best_model.pt')

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT = Path('/kaggle/input')
PRISTINE = INPUT / 'notebooks/itorousa/daps-pristine'
OUTPUT = Path('/kaggle/working')
STEP_DIR = OUTPUT / 'step_2a'
CHECKPOINT_DIR = STEP_DIR / 'checkpoints'
DATA_DIR = STEP_DIR / 'data'
MODEL_PATH = CHECKPOINT_DIR / 'best_model.pt'
STEP_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU — using CPU")
print(f"Device: {device}")

# ── Summary ───────────────────────────────────────────────────────────────────
total = N_CLIPS * DEGRADATIONS_PER_CLIP
print(f"\nStep 2a: Tier 1 Metric Literacy (EQ Degradation)")
print(f"Input:  81D (degraded metrics)")
print(f"Output: 81D (clean metrics)")
print(f"Degradation: EQ (random freq 20-18k Hz, gain ±12dB, Q={EQ_FIXED_Q})")
print(f"Clips: {N_CLIPS} ({CLIPS_PER_GENDER}m + {CLIPS_PER_GENDER}f)")
print(f"Degradations per clip: {DEGRADATIONS_PER_CLIP}")
print(f"Total samples: {total:,}")
print(f"Pretrained from: {PRETRAINED_PATH}")
