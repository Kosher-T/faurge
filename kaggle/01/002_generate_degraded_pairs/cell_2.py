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
N_PAIRS = 15000
PAIRS_PER_BATCH = 500
MAX_OUTPUT_GB = 19.0
random.seed(42)
np.random.seed(42)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Phase 1 outputs — attach ursula_cluster_data as Kaggle dataset
INPUT = Path('/kaggle/input')
CLUSTER_DATA = INPUT / 'notebooks/itorousa/01-cluster-speakers/ursula_cluster_data'

# Pristine audio
PRISTINE = {
    'vctk':    INPUT / 'notebooks/itorousa/vctk-pristine/pristine/wav48',
    'ljspeech': INPUT / 'notebooks/itorousa/ljspeech-pristine/pristine/wavs',
    'daps':    INPUT / 'notebooks/itorousa/daps-pristine',
}

OUTPUT = Path('/kaggle/working')
DATASET_DIR = OUTPUT / 'ursula_dataset'
PAIRS_DIR = DATASET_DIR / 'pairs'
DATASET_DIR.mkdir(parents=True, exist_ok=True)
PAIRS_DIR.mkdir(parents=True, exist_ok=True)

# ── Plugin Imports ────────────────────────────────────────────────────────────
# Kaggle utility scripts — each plugin is a separate script under /kaggle/usr/lib
PLUGIN_BASE = Path('/kaggle/usr/lib/notebooks/itorousa')
sys.path.insert(0, str(PLUGIN_BASE))

import compressor
import equalizer
import esser
import limiter
import saturator
import transient
import gain1