import json
import random
import shutil
import time
import zipfile
from collections import defaultdict
from pathlib import Path

# Set random seeds for reproducibility
random.seed(42)

# ─── Paths ───────────────────────────────────────────────────────────
DAPS_ENVIRONMENTS = [
    "iphone_livingroom1",
    "ipad_office2",
    "ipad_confroom2",
    "ipad_balcony1",
    "cleanraw",
]

# Kaggle input paths
DAPS_BASE_DIR = Path("/kaggle/input/datasets/itorousa/daps-audio-corpus/daps")
VCTK_BASE_DIR = Path("/kaggle/input/notebooks/itorousa/vctk-pristine/pristine/wav48")

# Output paths
OUTPUT_DIR = Path("/kaggle/working/daps_vctk_test")
OUTPUT_ZIP = Path("/kaggle/working/daps_vctk_test.zip")

print("Initializing download workspace...")
print(f"DAPS base: {DAPS_BASE_DIR}")
print(f"VCTK base: {VCTK_BASE_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
