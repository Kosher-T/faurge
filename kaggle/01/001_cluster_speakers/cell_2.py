!pip install pyloudnorm scikit-learn matplotlib seaborn

import json
import random
import time
from collections import defaultdict
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import soundfile as sf
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

random.seed(42)
np.random.seed(42)

# ─── Constants ───────────────────────────────────────────────────────
SR = 48000
CLIP_SEC = 5.0
CLIP_SAMPLES = int(SR * CLIP_SEC)  # 240,000
SAMPLES_PER_SPEAKER = 50
N_CLUSTERS = 8
BARK_N_BANDS = 64
BARK_LOW_HZ = 20.0
BARK_HIGH_HZ = 20000.0
FFT_SIZE = 4096
HOP_SIZE = 1024

# ─── Paths ───────────────────────────────────────────────────────────
PATHS = {
    "vctk": Path("/kaggle/input/notebooks/itorousa/vctk-pristine/pristine/wav48"),
    "ljspeech": Path("/kaggle/input/notebooks/itorousa/ljspeech-pristine/pristine/wavs"),
    "daps": Path("/kaggle/input/notebooks/itorousa/daps-pristine"),
}

OUTPUT = Path("/kaggle/working")
CLUSTER_DATA_DIR = OUTPUT / "ursula_cluster_data"
CLUSTER_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROFILES_CACHE = CLUSTER_DATA_DIR / "speaker_profiles.npz"

# ─── GPU Detection ───────────────────────────────────────────────────
try:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} — LTAS will use CUDA batch STFT")
    else:
        print("GPU: none detected — using CPU")
except ImportError:
    DEVICE = "cpu"
    print("PyTorch not installed — using CPU for all metrics")

print(f"SR={SR}, CLIP_SEC={CLIP_SEC}, CLIP_SAMPLES={CLIP_SAMPLES}")
print(f"Samples per speaker={SAMPLES_PER_SPEAKER}, Clusters={N_CLUSTERS}")
print(f"Output: {CLUSTER_DATA_DIR}")
