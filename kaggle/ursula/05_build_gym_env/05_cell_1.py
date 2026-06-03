# %% [markdown]
# # Phase 5: RL Gym Environment
#
# Creates the Gymnasium environment for training Ursula's DSP policy via SAC.
# The environment loads degraded/reference audio pairs and their 67D metrics,
# applies the 2-plugin cascade (EQ + Gain) from the policy's 125D action, and returns a
# reward based on how close the processed metrics are to the reference.
#
# **Inputs:** `ursula_raw_pairs/` (audio) + `ursula_metrics/` (metric tensors)
# **Outputs:** `UrsulaDSPEnv` — a Gymnasium env compatible with stable-baselines3 / SAC

import json, time, sys, os, csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces

# ── Constants ─────────────────────────────────────────────────────────────────
SR = 48000
CLIP_SEC = 5.0
CLIP_SAMPLES = int(SR * CLIP_SEC)  # 240,000
INPUT_DIM = 143       # 67 + 67 + 9
OUTPUT_DIM = 125      # EQ 124D + Gain 1D
N_CLUSTERS = 8
N_CLUSTERS_ONEHOT = N_CLUSTERS + 1  # +1 for "unknown"
METRIC_DIM = 67       # LTAS 64 + LUFS 1 + Crest 1 + ZCR 1
MAX_STEPS = 50        # max steps per episode
REWARD_MSE_MAX = 15000.0  # MSE ceiling mapped to -1 reward (narrow as she improves)

# ── Paths (Kaggle) ────────────────────────────────────────────────────────────
INPUT = Path('/kaggle/input')
PAIRS_DATA = INPUT / 'notebooks/itorousa/02-generate-degraded-pairs/ursula_dataset'
METRICS_DATA = INPUT / 'notebooks/itorousa/03-extract-metrics/ursula_metrics'
CLUSTER_DATA = INPUT / 'notebooks/itorousa/01-cluster-speakers/ursula_cluster_data'
OUTPUT = Path('/kaggle/working')

# ── Plugin path setup ─────────────────────────────────────────────────────────
# Add project root to path so we can import portable plugins
PROJECT_ROOT = Path('/kaggle/input/faurge-repo')  # Kaggle dataset with repo
if not PROJECT_ROOT.exists():
    # Fallback: try relative path (local dev)
    PROJECT_ROOT = Path(__file__).resolve().parents[4] if '__file__' in dir() else Path('.')
sys.path.insert(0, str(PROJECT_ROOT))

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"Pairs data: {PAIRS_DATA}")
print(f"Metrics data: {METRICS_DATA}")
