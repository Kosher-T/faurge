# ══════════════════════════════════════════════════════════════════════════════
# Step 1bc — Setup
# ══════════════════════════════════════════════════════════════════════════════
# Task: 160D input → 2D output (normalized log_freq, normalized gain)
# Model must learn to IDENTIFY frequency from metric pattern + predict gain.
# 10 clips (5m, 5f), 147 granular frequencies, 500 degradations each.

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# ── What are we doing? ───────────────────────────────────────────────────────
N_CLIPS = 10
CLIP_PREFIXES = ["m", "f"]
CLIPS_PER_GENDER = 5
DEGRADATIONS_PER_CLIP = 500

# Starting model (Step 1bc — knows 80D metrics, learns to reconstruct clean from degraded)
STEP1BC_MODEL = "/kaggle/input/notebooks/itorousa/ur-step-1bc/step_1bc/checkpoints/best_model.pt"

# Cache — set to path of saved .npz to skip generation, or None to generate
CACHE_PATH = None  # Set to path of saved .npz to skip generation

# Step 1bc normalizer — for normalizing 80D input metrics
STEP1BC_NORMALIZER = "/kaggle/input/notebooks/itorousa/ur-step-1bc/step_1bc/data/normalizer.json"

# Plugin paths (Kaggle cloud)
PLUGIN_DIR = Path('/kaggle/usr/lib/notebooks/itorousa')

# ── EQ settings ───────────────────────────────────────────────────────────────
EQ_FIXED_Q = 1.0
EQ_GAIN_RANGE_DB = (-12.0, 12.0)
EQ_FREQ_MIN_HZ = 20.0
EQ_FREQ_MAX_HZ = 18000.0

# ── 147 granular frequencies (non-round, log-spaced with jitter) ─────────────
FREQ_CURRICULUM = [
    20, 22, 23, 24, 25, 26, 28, 29, 31, 34,
    35, 36, 37, 39, 41, 44, 45, 47, 50, 51,
    54, 57, 60, 64, 69, 72, 73, 79, 81, 84,
    93, 97, 101, 102, 106, 115, 118, 121, 130, 132,
    146, 147, 158, 161, 171, 179, 184, 202, 209, 221,
    230, 237, 252, 253, 265, 275, 293, 307, 320, 346,
    352, 367, 390, 398, 434, 435, 481, 497, 502, 520,
    571, 594, 622, 653, 656, 698, 720, 788, 813, 836,
    862, 915, 959, 1028, 1070, 1137, 1161, 1190, 1291, 1355,
    1401, 1485, 1529, 1603, 1669, 1705, 1793, 1869, 2028, 2082,
    2205, 2364, 2379, 2514, 2687, 2725, 2826, 2996, 3112, 3411,
    3544, 3671, 3898, 4063, 4099, 4476, 4587, 4879, 5134, 5190,
    5365, 5656, 5991, 6420, 6700, 6737, 7228, 7523, 7783, 8097,
    8586, 9320, 9399, 9954, 10535, 10805, 11730, 12271, 12308, 13074,
    13524, 14143, 14585, 15800, 16432, 16740, 17763,
]

# ── Audio ─────────────────────────────────────────────────────────────────────
SR = 48000
CLIP_SEC = 5.0
CLIP_SAMPLES = int(SR * CLIP_SEC)

# ── Metrics ───────────────────────────────────────────────────────────────────
METRIC_DIM = 80  # LTAS64 + LUFS + Crest + ZCR + 13 Tier 1 spectral features
LTAS_DIM = 64
BARK_N_BANDS = 64
BARK_LOW_HZ = 20.0
BARK_HIGH_HZ = 20000.0
FFT_SIZE = 4096
HOP_SIZE = 1024

# ── Output normalization ─────────────────────────────────────────────────────
FREQ_LOG_MIN = np.log(EQ_FREQ_MIN_HZ)
FREQ_LOG_MAX = np.log(EQ_FREQ_MAX_HZ)
GAIN_MIN = EQ_GAIN_RANGE_DB[0]
GAIN_MAX = EQ_GAIN_RANGE_DB[1]

def normalize_freq(freq_hz):
    """Convert Hz to [0,1] via log scale."""
    return (np.log(freq_hz) - FREQ_LOG_MIN) / (FREQ_LOG_MAX - FREQ_LOG_MIN)

def denormalize_freq(norm):
    """Convert [0,1] back to Hz."""
    return np.exp(norm * (FREQ_LOG_MAX - FREQ_LOG_MIN) + FREQ_LOG_MIN)

def normalize_gain(gain_db):
    """Convert dB to [0,1]."""
    return (gain_db - GAIN_MIN) / (GAIN_MAX - GAIN_MIN)

def denormalize_gain(norm):
    """Convert [0,1] back to dB."""
    return norm * (GAIN_MAX - GAIN_MIN) + GAIN_MIN

# ── Training ──────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
LR = 5e-4
EPOCHS = 300
PATIENCE = 40
TRAIN_SPLIT = 0.8
HIDDEN_DIM = 128
FREQ_LOSS_WEIGHT = 0.5  # α for freq, (1-α) for gain

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT = Path('/kaggle/input')
PRISTINE = INPUT / 'notebooks/itorousa/daps-pristine'
OUTPUT = Path('/kaggle/working')
STEP1BC_DIR = OUTPUT / 'A_step_1bd'
CHECKPOINT_DIR = STEP1BC_DIR / 'checkpoints'
DATA_DIR = STEP1BC_DIR / 'data'
MODEL_PATH = CHECKPOINT_DIR / 'best_model.pt'
STEP1BC_DIR.mkdir(parents=True, exist_ok=True)
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
n_total = N_CLIPS * len(FREQ_CURRICULUM) * DEGRADATIONS_PER_CLIP
print(f"\nStep 1bc: Frequency + Gain Prediction")
print(f"Input:  160D (80 degraded + 80 clean metrics)")
print(f"Output: 2D (norm_log_freq, norm_gain)")
print(f"Metrics: LTAS64 + LUFS + Crest + ZCR + 13 Tier 1 spectral features")
print(f"Clips: {N_CLIPS} ({CLIPS_PER_GENDER}m + {CLIPS_PER_GENDER}f)")
print(f"Frequencies: {len(FREQ_CURRICULUM)}")
print(f"Degradations per clip per freq: {DEGRADATIONS_PER_CLIP}")
print(f"Total samples: {n_total:,}")
print(f"Step 1bc model: {STEP1BC_MODEL}")
