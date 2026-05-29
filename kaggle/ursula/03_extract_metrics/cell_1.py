import json, time, csv, gc
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch

# ── Constants ─────────────────────────────────────────────────────────────────
SR = 48000
CLIP_SEC = 5.0
CLIP_SAMPLES = int(SR * CLIP_SEC)  # 240,000
N_CLUSTERS = 8
BARK_N_BANDS = 64
BARK_LOW_HZ = 20.0
BARK_HIGH_HZ = 20000.0
FFT_SIZE = 4096
HOP_SIZE = 1024

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT = Path('/kaggle/input')

# Phase 1 outputs — cluster data
CLUSTER_DATA = INPUT / 'notebooks/itorousa/01-cluster-speakers/ursula_cluster_data'

# Phase 2 outputs — degraded pairs dataset
PAIRS_DATA = INPUT / 'notebooks/itorousa/02-generate-degraded-pairs/ursula_dataset'

OUTPUT = Path('/kaggle/working')
METRICS_DIR = OUTPUT / 'ursula_metrics'
METRICS_DIR.mkdir(parents=True, exist_ok=True)
PAIRS_OUT = METRICS_DIR / 'pairs'
PAIRS_OUT.mkdir(parents=True, exist_ok=True)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"Input pairs: {PAIRS_DATA}")
print(f"Output: {METRICS_DIR}")
