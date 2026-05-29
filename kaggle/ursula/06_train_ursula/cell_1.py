# %% [markdown]
# # Phase 6: RL Training (The Main Event)
#
# Trains Ursula's DSP policy via Soft Actor-Critic (SAC) on the Gymnasium
# environment built in Phase 5. The agent learns to map degraded/reference
# metric pairs → 227D plugin parameters that restore the audio toward the
# reference identity.
#
# **Inputs:** `ursula_metrics/` (metric tensors) + `ursula_raw_pairs/` (audio)
# **Outputs:** `ursula_v1.onnx` — trained policy exported to ONNX format

import json, time, sys, os, csv, random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import spaces

# ══════════════════════════════════════════════════════════════════════════════
# Import portable plugins (Kaggle top-level modules)
# ══════════════════════════════════════════════════════════════════════════════

PLUGIN_BASE = Path('/kaggle/usr/lib/notebooks/itorousa')
sys.path.insert(0, str(PLUGIN_BASE))

import compressor
import equalizer
import esser
import limiter
import saturator
import transient
import gain1

PLUGIN_MODULES = {
    'eq': equalizer,
    'compressor': compressor,
    'esser': esser,
    'saturator': saturator,
    'limiter': limiter,
    'transient': transient,
    'gain': gain1,
}

print("Loaded plugins:", list(PLUGIN_MODULES.keys()))

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

SR = 48000
CLIP_SEC = 5.0
CLIP_SAMPLES = int(SR * CLIP_SEC)
INPUT_DIM = 143       # 67 + 67 + 9
OUTPUT_DIM = 227      # all plugin params flattened
N_CLUSTERS = 8
N_CLUSTERS_ONEHOT = N_CLUSTERS + 1
METRIC_DIM = 67       # LTAS 64 + LUFS 1 + Crest 1 + ZCR 1
MAX_STEPS = 50

# ── Paths (Kaggle) ────────────────────────────────────────────────────────────
INPUT = Path('/kaggle/input')
PAIRS_DATA = INPUT / 'notebooks/itorousa/02-generate-degraded-pairs/ursula_dataset'
METRICS_DATA = INPUT / 'notebooks/itorousa/03-extract-metrics/ursula_metrics'
CLUSTER_DATA = INPUT / 'notebooks/itorousa/01-cluster-speakers/ursula_cluster_data'
OUTPUT = Path('/kaggle/working')
CHECKPOINT_DIR = OUTPUT / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ── Import agents.ursula from Phase 4 output dataset ────────────────────────
import shutil
AGENTS_SRC = Path('/kaggle/input/notebooks/itorousa/04-define-ursula/agents')
AGENTS_DST = OUTPUT / 'agents'
if AGENTS_SRC.exists():
    if AGENTS_DST.exists():
        shutil.rmtree(AGENTS_DST)
    shutil.copytree(AGENTS_SRC, AGENTS_DST)
    if not (AGENTS_DST / '__init__.py').exists():
        (AGENTS_DST / '__init__.py').write_text('')
    sys.path.insert(0, str(OUTPUT))

from agents.ursula import (
    UrsulaPolicy, UrsulaSACActor, UrsulaSACCritic,
    ActionUnnormalizer, ALL_PARAM_RANGES, CATEGORICAL_INDICES,
    PLUGIN_HEAD_ORDER, PLUGIN_HEAD_DIMS,
)

# ══════════════════════════════════════════════════════════════════════════════
# Reproducibility
# ══════════════════════════════════════════════════════════════════════════════

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"Seed: {SEED}")

# ══════════════════════════════════════════════════════════════════════════════
# SAC Hyperparameters
# ══════════════════════════════════════════════════════════════════════════════

REPLAY_BUFFER_SIZE = 500_000
BATCH_SIZE = 512
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4
GAMMA = 0.99            # discount
TAU = 0.005             # soft update
ALPHA_LR = 3e-4         # entropy auto-tune LR
WARMUP_STEPS = 5000     # random exploration before training
MAX_EPISODE_STEPS = 50
POLICY_DELAY = 2        # update actor every N critic steps

# ══════════════════════════════════════════════════════════════════════════════
# Curriculum Stages
# ══════════════════════════════════════════════════════════════════════════════

CURRICULUM = [
    {"name": "Phase A", "max_pairs": 3,    "start_step": 0,     "end_step": 20_000},
    {"name": "Phase B", "max_pairs": 30,   "start_step": 20_000, "end_step": 50_000},
    {"name": "Phase C", "max_pairs": None,  "start_step": 50_000, "end_step": float('inf')},
]

# ══════════════════════════════════════════════════════════════════════════════
# Logging state
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_LOG = {
    "rewards": [],
    "mse_values": [],
    "q_values": [],
    "alpha_values": [],
    "action_sparsity": [],
    "steps": [],
}

print(f"\n{'='*60}")
print(f"  PHASE 6: RL TRAINING")
print(f"{'='*60}")
print(f"  Buffer size:    {REPLAY_BUFFER_SIZE:,}")
print(f"  Batch size:     {BATCH_SIZE}")
print(f"  Actor LR:       {ACTOR_LR}")
print(f"  Critic LR:      {CRITIC_LR}")
print(f"  Gamma:          {GAMMA}")
print(f"  Tau:            {TAU}")
print(f"  Warmup steps:   {WARMUP_STEPS:,}")
print(f"  Policy delay:   {POLICY_DELAY}")
print(f"  Max episode:    {MAX_EPISODE_STEPS}")
print(f"  Curriculum:     {len(CURRICULUM)} stages")
for c in CURRICULUM:
    print(f"    {c['name']}: {c['max_pairs'] or 'full'} pairs, steps {c['start_step']:,}–{c['end_step']}")
print(f"{'='*60}\n")
