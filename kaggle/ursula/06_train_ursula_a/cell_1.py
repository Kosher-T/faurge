# %% [markdown]
# # Phase 6A: Inverse Degradation Supervised Pretraining
#
# Before RL, pretrain the policy network on degraded→reference pairs.
# For each pair, compute the approximate inverse of the degradation
# parameters (negate EQ gains), then train
# the policy to predict those inverse actions from metrics.
#
# This gives the RL agent a strong starting point — it begins with
# physically meaningful restoration parameters.
#
# **Inputs:** degradation_params.json + metrics + audio pairs
# **Outputs:** `ursula_pretrained.pt` — warm-start weights for RL

import json, time, sys, os, csv, random, shutil, math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Numba JIT ──
try:
    import numba
    print(f"Numba {numba.__version__} — JIT acceleration enabled")
except ImportError:
    numba = None
    print("Numba not available")

# ── Plugins ──
PLUGIN_BASE = Path('/kaggle/usr/lib/notebooks/itorousa')
sys.path.insert(0, str(PLUGIN_BASE))

import equalizer
import gain1

PLUGIN_MODULES = {
    'eq': equalizer, 'gain': gain1,
}
print("Loaded plugins:", list(PLUGIN_MODULES.keys()))

# ── Constants ──
SR = 48000
CLIP_SEC = 5.0
CLIP_SAMPLES = int(SR * CLIP_SEC)
INPUT_DIM = 143
OUTPUT_DIM = 188
N_CLUSTERS = 8
N_CLUSTERS_ONEHOT = N_CLUSTERS + 1
METRIC_DIM = 67

# ── Paths ──
INPUT = Path('/kaggle/input')
PAIRS_DATA = INPUT / 'notebooks/itorousa/02-generate-degraded-pairs/ursula_dataset'
METRICS_DATA = INPUT / 'notebooks/itorousa/03-extract-metrics/ursula_metrics'
CLUSTER_DATA = INPUT / 'notebooks/itorousa/01-cluster-speakers/ursula_cluster_data'
OUTPUT = Path('/kaggle/working')

AGENTS_SRC = INPUT / 'notebooks/itorousa/04-define-ursula/agents'
AGENTS_DST = OUTPUT / 'agents'
if AGENTS_SRC.exists():
    if AGENTS_DST.exists():
        shutil.rmtree(AGENTS_DST)
    shutil.copytree(AGENTS_SRC, AGENTS_DST)
    if not (AGENTS_DST / '__init__.py').exists():
        (AGENTS_DST / '__init__.py').write_text('')
    sys.path.insert(0, str(OUTPUT))

from agents.ursula import (
    UrsulaPolicy, UrsulaSACActor,
    ActionUnnormalizer, ALL_PARAM_RANGES, CATEGORICAL_INDICES,
    PLUGIN_HEAD_DIMS, PLUGIN_HEAD_ORDER,
)

# ── Reproducibility ──
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ── Hyperparameters ──
MAX_PAIRS = None                # use ALL pairs with degradation_params
SUPERVISED_EPOCHS = 500         # more epochs since we have good targets
SUPERVISED_BATCH_SIZE = 64
SUPERVISED_LR = 3e-4            # lower LR for cleaner convergence
AUGMENTATION_NOISE = 0.01       # smaller noise since targets are meaningful

print(f"\n{'='*60}")
print(f"  PHASE 6A: INVERSE DEGRADATION PRETRAINING")
print(f"{'='*60}")
print(f"  Pairs:           {'ALL' if MAX_PAIRS is None else MAX_PAIRS}")
print(f"  Epochs:          {SUPERVISED_EPOCHS}")
print(f"  Batch size:      {SUPERVISED_BATCH_SIZE}")
print(f"  LR:              {SUPERVISED_LR}")
print(f"  Aug noise:       {AUGMENTATION_NOISE}")
print(f"{'='*60}\n")
