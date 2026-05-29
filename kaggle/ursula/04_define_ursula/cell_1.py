# %% [markdown]
# # Phase 4: Ursula's Architecture
#
# Defines and validates Ursula's policy network for DSP parameter prediction.
# Input: 143D (M_degraded 67D || M_reference 67D || cluster_onehot 9D)
# Output: 227D (7 DSP plugins, tanh-activated, scaled to real ranges)
#
# **Inputs:** Metric tensors from Phase 3 (ursula_metrics/)
# **Outputs:** `agents/ursula.py` — UrsulaPolicy, UrsulaSACActor, UrsulaSACCritic

import math, json, time
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ─────────────────────────────────────────────────────────────────
SR = 48000
INPUT_DIM = 143       # 67 + 67 + 9
OUTPUT_DIM = 227      # all plugin params flattened
N_CLUSTERS = 8
N_CLUSTERS_ONEHOT = N_CLUSTERS + 1  # +1 for "unknown"
METRIC_DIM = 67       # LTAS 64 + LUFS 1 + Crest 1 + ZCR 1

# ── Paths (Kaggle) ────────────────────────────────────────────────────────────
INPUT = Path('/kaggle/input')
CLUSTER_DATA = INPUT / 'notebooks/itorousa/01-cluster-speakers/ursula_cluster_data'
METRICS_DATA = INPUT / 'notebooks/itorousa/03-extract-metrics/ursula_metrics'
OUTPUT = Path('/kaggle/working')

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")
print(f"Input dim: {INPUT_DIM}, Output dim: {OUTPUT_DIM}")
