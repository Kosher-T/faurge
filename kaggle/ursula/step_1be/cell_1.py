# ══════════════════════════════════════════════════════════════════════════════
# Step 1be — RMS Metric Literacy + Multi-Clip EQ Prediction: Setup
# ══════════════════════════════════════════════════════════════════════════════
# Phase 1: Metric literacy (162D → 81D). Degradation: EQ only (no global gain).
#   Input:  [degraded_81D, reference_81D] = 162D
#   Output: reference_81D
#   Teaches model what 81D metrics mean, including spectral changes from EQ.
#
# Phase 2: EQ prediction (162D → 3D). Degradation: 1 EQ band + global gain.
#   Input:  [degraded_81D, reference_81D] = 162D
#   Output: [freq, eq_gain, global_gain] = 3D
#   Initialized from Phase 1 weights. Multiple clips, multiple speakers.

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# ── What are we doing? ───────────────────────────────────────────────────────
N_TRAIN_CLIPS = 8
N_TEST_CLIPS = 2
DEGRADATIONS_PER_CLIP = 200
EVAL_DEGRADATIONS_PER_CLIP = 50

# ── Speakers ──────────────────────────────────────────────────────────────────
# DAPS speakers start with m/f (e.g. m001, f001).
# VCTK speakers are directories like p225, p226 (no gender tags).
# We scan both datasets and pick at random.
DAPS_DIR = Path('/kaggle/input/notebooks/itorousa/daps-pristine')
VCTK_DIR = Path('/kaggle/input/notebooks/itorousa/vctk-pristine/pristine/wav48')

# ── Plugin paths (Kaggle cloud) ──────────────────────────────────────────────
PLUGIN_DIR = Path('/kaggle/usr/lib/notebooks/itorousa')

# ── Cache — upload completed datasets as Kaggle datasets, set paths here ────
# Set to None to generate from scratch. Set to a path to load from cache.
PHASE1_CACHE = None  # e.g. '/kaggle/input/step1be-phase1/phase1_dataset.npz'
PHASE2_CACHE = None  # e.g. '/kaggle/input/step1be-phase2/phase2_dataset.npz'
# Phase 2 partial — upload partial .npz as Kaggle dataset to resume across sessions
PHASE2_PARTIAL_LOAD = None  # e.g. '/kaggle/input/step1be-partial/data/phase2_partial.npz'
# Eval datasets are resumable via partial saves (no manual upload needed)

# ── EQ settings ──────────────────────────────────────────────────────────────
EQ_FIXED_Q = 1.0
EQ_GAIN_RANGE_DB = (-12.0, 12.0)
EQ_FREQ_MIN_HZ = 20.0
EQ_FREQ_MAX_HZ = 18000.0

# ── Gain settings ────────────────────────────────────────────────────────────
GAIN_RANGE_DB = (-6.0, 6.0)  # global gain range

# ── 147 granular frequencies (non-round, log-spaced with jitter) ────────────
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

# ── Frequency split for generalization testing ──────────────────────────────
UNSEEN_FREQS = FREQ_CURRICULUM[::6]          # every 6th → 27 frequencies
TRAIN_FREQS = [f for f in FREQ_CURRICULUM if f not in UNSEEN_FREQS]  # 120

# ── Audio ────────────────────────────────────────────────────────────────────
SR = 48000
CLIP_SEC = 5.0
CLIP_SAMPLES = int(SR * CLIP_SEC)

# ── Metrics ──────────────────────────────────────────────────────────────────
METRIC_DIM = 81  # 80D (Tier 0+1) + RMS Energy
LTAS_DIM = 64
BARK_LOW_HZ = 20.0
BARK_HIGH_HZ = 20000.0
FFT_SIZE = 4096
HOP_SIZE = 1024

# ── Output normalization (Phase 2) ──────────────────────────────────────────
FREQ_LOG_MIN = np.log(EQ_FREQ_MIN_HZ)
FREQ_LOG_MAX = np.log(EQ_FREQ_MAX_HZ)
EQ_GAIN_MIN = EQ_GAIN_RANGE_DB[0]
EQ_GAIN_MAX = EQ_GAIN_RANGE_DB[1]
GAIN_MIN = GAIN_RANGE_DB[0]
GAIN_MAX = GAIN_RANGE_DB[1]

def normalize_freq(freq_hz):
    return (np.log(freq_hz) - FREQ_LOG_MIN) / (FREQ_LOG_MAX - FREQ_LOG_MIN)

def denormalize_freq(norm):
    return np.exp(norm * (FREQ_LOG_MAX - FREQ_LOG_MIN) + FREQ_LOG_MIN)

def normalize_eq_gain(gain_db):
    return (gain_db - EQ_GAIN_MIN) / (EQ_GAIN_MAX - EQ_GAIN_MIN)

def denormalize_eq_gain(norm):
    return norm * (EQ_GAIN_MAX - EQ_GAIN_MIN) + EQ_GAIN_MIN

def normalize_global_gain(gain_db):
    return (gain_db - GAIN_MIN) / (GAIN_MAX - GAIN_MIN)

def denormalize_global_gain(norm):
    return norm * (GAIN_MAX - GAIN_MIN) + GAIN_MIN

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 200
PATIENCE = 30
TRAIN_SPLIT = 0.8
HIDDEN_DIM = 128

# Phase 2 loss weights (freq + eq_gain weighted higher than global_gain)
FREQ_LOSS_WEIGHT = 0.35
EQ_GAIN_LOSS_WEIGHT = 0.35
GLOBAL_GAIN_LOSS_WEIGHT = 0.30

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT = Path('/kaggle/input')
OUTPUT = Path('/kaggle/working')
STEP_DIR = OUTPUT / 'step_1be'
CHECKPOINT_DIR = STEP_DIR / 'checkpoints'
DATA_DIR = STEP_DIR / 'data'
PHASE1_MODEL_PATH = CHECKPOINT_DIR / 'phase1_best_model.pt'
PHASE1_ENCODER_PATH = CHECKPOINT_DIR / 'phase1_encoder.pt'
PHASE2_MODEL_PATH = CHECKPOINT_DIR / 'phase2_best_model.pt'
NORMALIZER_PATH = DATA_DIR / 'normalizer_81d.json'
STEP_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Device ───────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU — using CPU")
print(f"Device: {device}")

# ── Summary ──────────────────────────────────────────────────────────────────
total = (N_TRAIN_CLIPS + N_TEST_CLIPS) * len(FREQ_CURRICULUM) * DEGRADATIONS_PER_CLIP
print(f"\nStep 1be: RMS Metric Literacy + Multi-Clip EQ Prediction")
print(f"{'='*60}")
print(f"Metric dim: {METRIC_DIM}D (67D Tier 0 + 13D Tier 1 + 1D RMS)")
print(f"Train clips: {N_TRAIN_CLIPS} | Test clips: {N_TEST_CLIPS}")
print(f"Train freqs: {len(TRAIN_FREQS)} | Unseen freqs: {len(UNSEEN_FREQS)}")
print(f"Degradations per clip: {DEGRADATIONS_PER_CLIP}")
print(f"Total samples: {total:,}")
print()
print("PHASE 1 — Metric Literacy (EQ degradation)")
print(f"  Input:  162D = [degraded_81D, reference_81D]")
print(f"  Output:  81D = reference_81D")
print(f"  What:    Take clean audio, apply 1 EQ band at random")
print(f"           frequency (from {len(TRAIN_FREQS)}-freq grid) with random gain")
print(f"           (±12dB). Model learns to reconstruct the original")
print(f"           metrics from the degraded+reference pair.")
print(f"  Why:     EQ changes Tier 1 features (spectral shape).")
print(f"           Gain-only doesn't — model would learn nothing.")
print(f"           This teaches the model what spectral changes look like.")
print()
print("PHASE 2 — EQ Prediction (EQ + gain degradation)")
print(f"  Input:  162D = [degraded_81D, reference_81D]")
print(f"  Output:   3D = [freq, eq_gain, global_gain]")
print(f"  What:    Take clean audio, apply 1 EQ band at random")
print(f"           frequency (from {len(TRAIN_FREQS)}-freq grid) with random gain")
print(f"           (±12dB), plus random global gain (±6dB).")
print(f"           Model predicts the EQ frequency, EQ gain,")
print(f"           and global gain from the degraded+reference pair.")
print(f"  Why:     Multiple clips + RMS metric should let the model")
print(f"           separate EQ shape from global level changes.")
print(f"{'='*60}")
