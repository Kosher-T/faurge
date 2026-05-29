import json, random, time, os, sys, gc
from pathlib import Path
from collections import defaultdict
import numpy as np
import soundfile as sf
import librosa

# ── Constants ─────────────────────────────────────────────────────────────────
SR = 48000
CLIP_SEC = 5.0
CLIP_SAMPLES = int(SR * CLIP_SEC)  # 240,000
CROSSFADE_MS = 500
CROSSFADE_SAMPLES = int(SR * CROSSFADE_MS / 1000)

# Per-dataset clip limits
VCTK_MAX_CLIPS_PER_SPEAKER = 120
LJSpeech_MAX_CLIPS = 4000
DAPS_MAX_CLIPS = None  # all

# Metric extraction constants (matching Phase 1)
N_CLUSTERS = 8
BARK_N_BANDS = 64
BARK_LOW_HZ = 20.0
BARK_HIGH_HZ = 20000.0
FFT_SIZE = 4096
HOP_SIZE = 1024

PAIRS_PER_BATCH = 500
MAX_OUTPUT_GB = 18.0
CHECKPOINT_INTERVAL = 100

random.seed(42)
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
INPUT = Path('/kaggle/input')

# Phase 1 outputs — attach ursula_cluster_data as Kaggle dataset
CLUSTER_DATA = INPUT / 'notebooks/itorousa/01-cluster-speakers/ursula_cluster_data'

# Pristine audio
PRISTINE = {
    'vctk':     INPUT / 'notebooks/itorousa/vctk-pristine/pristine/wav48',
    'ljspeech': INPUT / 'notebooks/itorousa/ljspeech-pristine/pristine/wavs',
    'daps':     INPUT / 'notebooks/itorousa/daps-pristine',
}

OUTPUT = Path('/kaggle/working')
DATASET_DIR = OUTPUT / 'ursula_dataset'
PAIRS_DIR = DATASET_DIR / 'pairs'
DATASET_DIR.mkdir(parents=True, exist_ok=True)
PAIRS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plugin Imports ────────────────────────────────────────────────────────────
PLUGIN_BASE = Path('/kaggle/usr/lib/notebooks/itorousa')
sys.path.insert(0, str(PLUGIN_BASE))

import compressor
import equalizer
import esser
import limiter
import saturator
import transient
import gain1
