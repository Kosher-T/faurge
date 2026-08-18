# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Causality: Setup
# ══════════════════════════════════════════════════════════════════════════════
# Teach Ursula that tweaking one plugin parameter affects specific metrics.
# Input:  degraded metrics (67D) + clean metrics (67D) = 134D
# Output: plugin parameter(s) to fix the audio
#
# Substeps:
#   1a: 1 clip,   gain only          → predict gain_db (1D)
#   1b: 5 clips,  gain only          → predict gain_db (1D)
#   1c: 1 clip,   EQ only (1 band)   → predict freq, gain, q (3D)
#   1d: 5 clips,  EQ only (1 band)   → predict freq, gain, q (3D)
#   1e: 1 clip,   gain + EQ (1 band) → predict gain_db + freq, gain, q (4D)
#   1f: 5 clips,  gain + EQ (1 band) → predict gain_db + freq, gain, q (4D)

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# ── Substep Config ────────────────────────────────────────────────────────────
# Change SUBSTEP and LOAD_MODEL_FROM, everything else follows.
SUBSTEP = "1a"

# Plugin selection
USE_GAIN = True
USE_EQ = False
USE_COMPRESSOR = False
N_EQ_BANDS = 1       # How many EQ bands to predict

# Clip selection
N_CLIPS = 1          # 1a,1c,1e: 1 clip. 1b,1d,1f: 5 clips.
CLIP_PREFIXES = ["m", "f"]  # Mix male and female. Use ["m"] or ["f"] for single gender.
DEGRADATIONS_PER_CLIP = 200

# Model path (previous step's output)
LOAD_MODEL_FROM = "/kaggle/input/notebooks/itorousa/ur-step-0/step_0/checkpoints/best_model.pt"

# ── Plugin paths (Kaggle cloud) ───────────────────────────────────────────────
PLUGIN_DIR = Path('/kaggle/usr/lib/notebooks/itorousa')

# ── Plugin ranges (linear only — nonlinear belongs to Genesis) ───────────────
# Gain
GAIN_RANGE_DB = (-12.0, 12.0)

# EQ (parametric, random bands)
EQ_GAIN_RANGE_DB = (-6.0, 6.0)
EQ_FREQ_RANGE_HZ = (80.0, 8000.0)
EQ_Q_RANGE = (0.5, 3.0)

# Compressor
COMP_THRESHOLD_RANGE_DB = (-40.0, -10.0)
COMP_RATIO_RANGE = (1.0, 8.0)
COMP_ATTACK_RANGE_MS = (1.0, 20.0)
COMP_RELEASE_RANGE_MS = (50.0, 300.0)

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

# ── Compute output dim from plugin config ─────────────────────────────────────
OUTPUT_DIM = 0
if USE_GAIN:
    OUTPUT_DIM += 1
if USE_EQ:
    OUTPUT_DIM += 3 * N_EQ_BANDS
if USE_COMPRESSOR:
    OUTPUT_DIM += 4

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 500
PATIENCE = 100
TRAIN_SPLIT = 0.8
HIDDEN_DIM = 128

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT = Path('/kaggle/input')
PRISTINE = INPUT / 'notebooks/itorousa/daps-pristine'
OUTPUT = Path('/kaggle/working')
STEP1_DIR = OUTPUT / 'step_1'
CHECKPOINT_DIR = STEP1_DIR / 'checkpoints'
MODEL_PATH = CHECKPOINT_DIR / 'best_model.pt'
STEP1_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU — using CPU (training will be slow)")
print(f"Device: {device}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\nSubstep: {SUBSTEP}")
print(f"Plugins: gain={USE_GAIN}, eq={USE_EQ}, compressor={USE_COMPRESSOR}")
print(f"EQ bands: {N_EQ_BANDS}")
print(f"Clips: {N_CLIPS} (prefixes={CLIP_PREFIXES})")
print(f"Input dim: {METRIC_DIM * 2} = degraded({METRIC_DIM}) + clean({METRIC_DIM})")
print(f"Output dim: {OUTPUT_DIM}")
print(f"Load model from: {LOAD_MODEL_FROM}")