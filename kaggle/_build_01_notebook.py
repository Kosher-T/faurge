#!/usr/bin/env python3
"""
Generates kaggle/01_acquire_and_augment.ipynb using nbformat.
Run once locally:  python kaggle/_build_notebook.py
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
nb.metadata.kernelspec = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}

# ======================================================================
# CELL 1 — Title
# ======================================================================
nb.cells.append(nbf.v4.new_markdown_cell("""\
# 🧬 Genesis IR Head — Data Curation Pipeline

**Notebook 01**: Acquire → Sterilize → Augment → Target

This notebook produces training triples for Genesis's IR Synthesis Head:

| Tensor | Format | Description |
|--------|--------|-------------|
| `source_wet` | int16 waveform | Sterile vocal + bad-room IR + random degradations |
| `target_wet` | int16 waveform | Same sterile vocal + studio IR (ground truth) |
| `target_clap` | float32 vector | CLAP embedding of the target-wet audio |

**Checkpoint chaining**: Output is capped at ~19 GB. A `checkpoint.json` tracks
progress so the next run (using this output as input) continues where we left off.
"""))

# ======================================================================
# CELL 2 — Installs
# ======================================================================
nb.cells.append(nbf.v4.new_code_cell("""\
!pip install -q noisereduce pyloudnorm soundfile librosa transformers torch
"""))

# ======================================================================
# CELL 3 — Imports & Configuration
# ======================================================================
nb.cells.append(nbf.v4.new_code_cell("""\
import os, json, hashlib, time, random, warnings, struct
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

import numpy as np
import librosa
import soundfile as sf
from scipy.signal import fftconvolve
import noisereduce as nr

warnings.filterwarnings('ignore')
random.seed(42)
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════
# KAGGLE INPUT PATHS  (datasets attached to this notebook)
# ═══════════════════════════════════════════════════════════════════════
PATHS = {
    'irs':       Path('/kaggle/input/impulse-responses'),
    'ljspeech':  Path('/kaggle/input/ljspeech'),
    'vctk':      Path('/kaggle/input/vctk-corpus/VCTK-Corpus/wav48'),
    'langid_en': Path('/kaggle/input/language-identifier/english/clips'),
}

# If chaining from a previous run, attach its output as input here:
PREV_RUN_PATH = Path('/kaggle/input/genesis-data-run1')  # adjust per run

# ═══════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════
OUTPUT       = Path('/kaggle/working')
BATCH_DIR    = OUTPUT / 'batches'
ACQUIRED_DIR = OUTPUT / 'acquired_irs'
CLAP_DIR     = OUTPUT / 'clap_model'

for d in [BATCH_DIR, ACQUIRED_DIR, CLAP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# AUDIO PARAMETERS
# ═══════════════════════════════════════════════════════════════════════
SR            = 48000
CLIP_SEC      = 5.0
CLIP_SAMPLES  = int(SR * CLIP_SEC)    # 240,000 samples

# ═══════════════════════════════════════════════════════════════════════
# BUDGET & BATCHING
# ═══════════════════════════════════════════════════════════════════════
TRIPLES_PER_BATCH  = 500
MAX_OUTPUT_GB      = 19.0             # safety margin under 20 GB cap
AUGMENTATIONS_MIN  = 3
AUGMENTATIONS_MAX  = 6

print(f"SR={SR}, CLIP_SEC={CLIP_SEC}, CLIP_SAMPLES={CLIP_SAMPLES}")
print(f"Output budget: {MAX_OUTPUT_GB} GB, batch size: {TRIPLES_PER_BATCH}")
"""))

# ======================================================================
# CELL 4 — Checkpoint System
# ======================================================================
nb.cells.append(nbf.v4.new_code_cell("""\
# ═══════════════════════════════════════════════════════════════════════
# CHECKPOINT  — resume across runs
# ═══════════════════════════════════════════════════════════════════════
CHECKPOINT_PATH = OUTPUT / 'checkpoint.json'

def load_checkpoint() -> dict:
    \"\"\"Load from previous run's output (if chained) or current working dir.\"\"\"
    # Check previous run first
    prev_ckpt = PREV_RUN_PATH / 'checkpoint.json'
    if prev_ckpt.exists():
        with open(prev_ckpt) as f:
            ckpt = json.load(f)
        ckpt['run_number'] += 1
        print(f"[Checkpoint] Resuming from previous run: {ckpt['triples_completed']} triples done")
        return ckpt

    # Check current working dir
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)

    # Fresh start
    return {
        'batch_id': 0,
        'triples_completed': 0,
        'vocal_cursor': 0,        # index into shuffled vocal list
        'run_number': 1,
    }

def save_checkpoint(ckpt: dict):
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(ckpt, f, indent=2)

def get_output_size_gb() -> float:
    total = sum(f.stat().st_size for f in OUTPUT.rglob('*') if f.is_file())
    return total / (1024 ** 3)

ckpt = load_checkpoint()
print(f"[Checkpoint] Run #{ckpt['run_number']}, "
      f"{ckpt['triples_completed']} triples completed so far, "
      f"starting at vocal cursor {ckpt['vocal_cursor']}")
"""))

# ======================================================================
# CELL 5 — Acquire & Normalize IRs
# ======================================================================
nb.cells.append(nbf.v4.new_code_cell("""\
import urllib.request, zipfile

# ═══════════════════════════════════════════════════════════════════════
# STAGE 1: ACQUIRE — download external IR collections
# ═══════════════════════════════════════════════════════════════════════

def download_if_needed(url: str, dest_dir: Path, name: str):
    marker = dest_dir / f'.{name}_done'
    if marker.exists():
        print(f"  [Acquire] {name} already downloaded")
        return
    print(f"  [Acquire] Downloading {name}...")
    zip_path = dest_dir / f'{name}.zip'
    urllib.request.urlretrieve(url, str(zip_path))
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir / name)
    zip_path.unlink()
    marker.touch()
    print(f"  [Acquire] {name} ✓")

# MIT Impulse Response Survey — 271 real-world room IRs (CC-BY 4.0)
download_if_needed(
    'https://mcdermottlab.mit.edu/Reverb/IRMAudio/Audio.zip',
    ACQUIRED_DIR, 'mit_irs'
)

# ───────────────────────────────────────────────────────────────────────
# Load & normalize every IR to 48 kHz mono
# ───────────────────────────────────────────────────────────────────────

def load_and_normalize_ir(filepath: Path, target_sr: int = SR) -> Optional[np.ndarray]:
    \"\"\"Load an IR file, force mono / target SR, peak-normalize.\"\"\"
    try:
        audio, _ = librosa.load(str(filepath), sr=target_sr, mono=True)
        if len(audio) < 64:        # degenerate
            return None
        peak = np.max(np.abs(audio))
        if peak > 1e-6:
            audio = audio / peak
        return audio.astype(np.float32)
    except Exception:
        return None

print("\\n[Acquire] Loading all impulse responses...")
all_irs: Dict[str, dict] = {}

# 1) User's IRs  (supports .irs and .wav)
for ext in ('*.irs', '*.wav'):
    for f in sorted(PATHS['irs'].glob(ext)):
        ir = load_and_normalize_ir(f)
        if ir is not None:
            all_irs[f'user_{f.stem}'] = {'audio': ir, 'source': 'user'}

# 2) MIT IRs
mit_dir = ACQUIRED_DIR / 'mit_irs'
if mit_dir.exists():
    for f in sorted(mit_dir.rglob('*.wav')):
        ir = load_and_normalize_ir(f)
        if ir is not None:
            all_irs[f'mit_{f.stem}'] = {'audio': ir, 'source': 'mit'}

print(f"[Acquire] Loaded {len(all_irs)} IRs "
      f"(user: {sum(1 for v in all_irs.values() if v['source']=='user')}, "
      f"MIT: {sum(1 for v in all_irs.values() if v['source']=='mit')})")
"""))

# ======================================================================
# CELL 6 — Classify IRs into Bad / Studio pools
# ======================================================================
nb.cells.append(nbf.v4.new_code_cell("""\
# ═══════════════════════════════════════════════════════════════════════
# Classify IRs into "bad" (room/degraded) and "studio" (clean target)
# ═══════════════════════════════════════════════════════════════════════

def compute_ir_features(ir_audio: np.ndarray, sr: int = SR) -> dict:
    \"\"\"Compute RT60 estimate, spectral centroid, and clarity (C50).\"\"\"
    energy = ir_audio ** 2
    cumsum = np.cumsum(energy[::-1])[::-1]

    # RT60 — time for energy to decay 60 dB
    rt60 = len(ir_audio) / sr
    if cumsum[0] > 1e-10:
        decay_db = 10 * np.log10(cumsum / cumsum[0] + 1e-12)
        idx = np.where(decay_db < -60)[0]
        if len(idx) > 0:
            rt60 = idx[0] / sr

    # Spectral centroid
    S = np.abs(np.fft.rfft(ir_audio))
    freqs = np.fft.rfftfreq(len(ir_audio), 1 / sr)
    centroid = float(np.sum(freqs * S) / (np.sum(S) + 1e-10))

    # Clarity C50 — early-to-late energy ratio at 50 ms boundary
    split = int(0.05 * sr)
    early = np.sum(ir_audio[:split] ** 2) + 1e-12
    late  = np.sum(ir_audio[split:] ** 2) + 1e-12
    c50   = float(10 * np.log10(early / late))

    return {
        'rt60': round(rt60, 4),
        'centroid': round(centroid, 1),
        'c50': round(c50, 2),
        'length_sec': round(len(ir_audio) / sr, 4),
    }

ir_catalogue = {}
bad_pool:    List[str] = []   # contamination IRs (rooms, halls, degraded)
studio_pool: List[str] = []   # target IRs (mastering, clarity, studio)

for ir_id, ir_data in all_irs.items():
    feats = compute_ir_features(ir_data['audio'])
    ir_data['features'] = feats
    ir_catalogue[ir_id] = {'source': ir_data['source'], **feats}

    # Heuristic:
    #   MIT IRs → always bad pool (real rooms)
    #   User IRs with long RT60 or low clarity → bad pool
    #   User IRs with short RT60 and high clarity → studio pool
    if ir_data['source'] == 'mit':
        bad_pool.append(ir_id)
    elif feats['rt60'] > 0.25 or feats['c50'] < 8:
        bad_pool.append(ir_id)
    else:
        studio_pool.append(ir_id)

# Safety: ensure both pools are adequate
if len(studio_pool) < 20:
    # Sort user IRs by clarity, take top third as studio
    user_ids = sorted(
        [k for k, v in all_irs.items() if v['source'] == 'user'],
        key=lambda k: all_irs[k]['features']['c50'], reverse=True
    )
    studio_pool = user_ids[:max(50, len(user_ids) // 3)]
    bad_pool = [k for k in all_irs if k not in studio_pool]

print(f"[Classify] Bad pool: {len(bad_pool)} IRs")
print(f"[Classify] Studio pool: {len(studio_pool)} IRs")

# Save catalogue
with open(OUTPUT / 'ir_catalogue.json', 'w') as f:
    json.dump(ir_catalogue, f, indent=2)
print("[Classify] ir_catalogue.json saved ✓")
"""))

# ======================================================================
# CELL 7 — Freeze CLAP & Pre-compute Studio Embeddings
# ======================================================================
nb.cells.append(nbf.v4.new_code_cell("""\
import torch
from transformers import ClapModel, ClapProcessor

# ═══════════════════════════════════════════════════════════════════════
# STAGE 2: Freeze CLAP model & cache studio-pool embeddings
# ═══════════════════════════════════════════════════════════════════════

CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[CLAP] Device: {device}")

print("[CLAP] Loading model...")
clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
clap_model = ClapModel.from_pretrained(CLAP_MODEL_ID).to(device).eval()
CLAP_DIM = clap_model.config.projection_dim
print(f"[CLAP] Loaded — embedding dim = {CLAP_DIM}")

# Save frozen model for downstream notebooks
clap_model.save_pretrained(CLAP_DIR)
clap_processor.save_pretrained(CLAP_DIR)
print(f"[CLAP] Frozen model saved to {CLAP_DIR}")


def get_clap_audio_embedding(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    \"\"\"Encode audio through CLAP's audio tower. Returns (CLAP_DIM,) float32.\"\"\"
    inputs = clap_processor(
        audios=audio, sampling_rate=sr, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        emb = clap_model.get_audio_features(**inputs)
    return emb.cpu().numpy().flatten().astype(np.float32)


# ───────────────────────────────────────────────────────────────────
# Pre-compute CLAP embeddings for every studio-pool IR.
# We convolve each IR with 3 seconds of white noise so CLAP has
# a richer scene to analyse (raw IRs are too short / sparse).
# ───────────────────────────────────────────────────────────────────
print(f"\\n[CLAP] Pre-computing embeddings for {len(studio_pool)} studio IRs...")
ref_noise = np.random.randn(SR * 3).astype(np.float32) * 0.1

clap_cache: Dict[str, np.ndarray] = {}
for i, ir_id in enumerate(studio_pool):
    ir_audio = all_irs[ir_id]['audio']
    scene = fftconvolve(ref_noise, ir_audio, mode='full')[:SR * 3]
    scene = scene / (np.max(np.abs(scene)) + 1e-8)
    clap_cache[ir_id] = get_clap_audio_embedding(scene)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(studio_pool)}")

np.savez(OUTPUT / 'clap_cache.npz', **clap_cache)
print(f"[CLAP] Cached {len(clap_cache)} embeddings ✓  (dim={CLAP_DIM})")
"""))

# ======================================================================
# CELL 8 — Sterilize Vocals
# ======================================================================
nb.cells.append(nbf.v4.new_code_cell("""\
import pyloudnorm as pyln

# ═══════════════════════════════════════════════════════════════════════
# STAGE 3: STERILIZE — noise reduce, trim, segment, normalize
# ═══════════════════════════════════════════════════════════════════════

meter = pyln.Meter(SR)

def discover_audio_files() -> List[Tuple[Path, str]]:
    \"\"\"Collect all vocal files from all datasets. Returns (path, dataset_tag).\"\"\"
    files = []
    # LJSpeech
    for f in sorted(PATHS['ljspeech'].rglob('*.wav')):
        files.append((f, 'ljspeech'))
    # VCTK — subfolder per speaker
    for f in sorted(PATHS['vctk'].rglob('*.wav')):
        files.append((f, f'vctk_{f.parent.name}'))
    # Language Identifier — english clips
    for ext in ('*.wav', '*.mp3', '*.ogg'):
        for f in sorted(PATHS['langid_en'].rglob(ext)):
            files.append((f, 'langid_en'))
    return files


def sterilize_and_segment(filepath: Path, tag: str) -> List[dict]:
    \"\"\"
    Load → noise-reduce → trim silence → segment into CLIP_SEC windows.
    Returns list of dicts with 'audio' (float32) and metadata.
    \"\"\"
    try:
        audio, sr_orig = librosa.load(str(filepath), sr=SR, mono=True)
    except Exception:
        return []

    if len(audio) < SR * 1.0:   # skip clips shorter than 1 second
        return []

    # ── Spectral noise reduction (simulates Ursula's output) ──
    audio = nr.reduce_noise(y=audio, sr=SR, stationary=True, prop_decrease=0.85)

    # ── Trim silence ──
    audio, _ = librosa.effects.trim(audio, top_db=40)
    if len(audio) < SR * 1.5:
        return []

    # ── Normalize loudness to -23 LUFS ──
    try:
        loudness = meter.integrated_loudness(audio)
        if loudness > -70:  # not silence
            audio = pyln.normalize.loudness(audio, loudness, -23.0)
    except Exception:
        pass

    # ── Segment into CLIP_SEC windows ──
    segments = []
    for start in range(0, len(audio) - SR, CLIP_SAMPLES):
        chunk = audio[start : start + CLIP_SAMPLES]
        if len(chunk) < CLIP_SAMPLES:
            chunk = np.pad(chunk, (0, CLIP_SAMPLES - len(chunk)))

        # Skip near-silent segments
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms < 1e-4:
            continue

        segments.append({
            'audio':   chunk.astype(np.float32),
            'file':    filepath.name,
            'dataset': tag,
        })
    return segments


# ── Discover & sterilize ──
print("[Sterilize] Discovering audio files...")
all_audio_files = discover_audio_files()
print(f"[Sterilize] Found {len(all_audio_files)} source files")

# Shuffle deterministically so runs are balanced across datasets
random.shuffle(all_audio_files)

# Process in batches to manage memory
STERILIZE_CHUNK = 500   # files processed at a time
vocal_segments: List[dict] = []
cursor_start = ckpt.get('vocal_cursor', 0) // 5  # approximate file index

print(f"[Sterilize] Processing from file ~{cursor_start}...")
for i, (fpath, tag) in enumerate(all_audio_files):
    if i % 200 == 0 and i > 0:
        print(f"  Processed {i}/{len(all_audio_files)} files → "
              f"{len(vocal_segments)} segments so far")
    segs = sterilize_and_segment(fpath, tag)
    vocal_segments.extend(segs)

print(f"\\n[Sterilize] Total sterile segments: {len(vocal_segments)}")
print(f"[Sterilize] Dataset breakdown:")
ds_counts = defaultdict(int)
for s in vocal_segments:
    ds_counts[s['dataset'].split('_')[0]] += 1
for ds, cnt in sorted(ds_counts.items()):
    print(f"  {ds}: {cnt}")
"""))

# ======================================================================
# CELL 9 — Noise & Degradation Toolkit
# ======================================================================
nb.cells.append(nbf.v4.new_code_cell("""\
# ═══════════════════════════════════════════════════════════════════════
# DEGRADATION TOOLKIT — applied to source_wet ONLY
# ═══════════════════════════════════════════════════════════════════════

def add_noise(audio: np.ndarray, noise_type: str, snr_db: float) -> np.ndarray:
    \"\"\"Mix coloured noise at the target SNR.\"\"\"
    n = len(audio)
    if noise_type == 'white':
        noise = np.random.randn(n)
    elif noise_type == 'pink':
        # Pink noise via spectral shaping
        freqs = np.fft.rfftfreq(n, 1 / SR)
        freqs[0] = 1
        S = 1.0 / np.sqrt(freqs)
        noise = np.fft.irfft(S * np.exp(2j * np.pi * np.random.rand(len(S))))[:n]
    elif noise_type == 'brown':
        noise = np.cumsum(np.random.randn(n))
        noise -= np.mean(noise)
    elif noise_type == 'hvac':
        # Band-limited broadband 100-1000 Hz
        noise = np.random.randn(n)
        from scipy.signal import butter, sosfilt
        sos = butter(4, [100, 1000], btype='band', fs=SR, output='sos')
        noise = sosfilt(sos, noise)
    elif noise_type == 'hum':
        # 50 Hz fundamental + harmonics
        t = np.arange(n) / SR
        base_freq = random.choice([50, 60])
        noise = np.zeros(n)
        for h in range(1, 6):
            amp = 1.0 / h
            noise += amp * np.sin(2 * np.pi * base_freq * h * t + random.uniform(0, 2*np.pi))
    else:
        noise = np.random.randn(n)

    # Scale noise to target SNR
    sig_power = np.mean(audio ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    return audio + noise.astype(np.float32)


def apply_eq(audio: np.ndarray) -> np.ndarray:
    \"\"\"Random 3-band parametric EQ (simulates mic coloration).\"\"\"
    from scipy.signal import butter, sosfilt
    bands = [
        (80, 300),     # low
        (300, 3000),   # mid
        (3000, 12000), # high
    ]
    for lo, hi in bands:
        gain_db = random.uniform(-6, 6)
        if abs(gain_db) < 1:
            continue
        try:
            sos = butter(2, [lo, hi], btype='band', fs=SR, output='sos')
            band_sig = sosfilt(sos, audio)
            gain_lin = 10 ** (gain_db / 20)
            audio = audio + band_sig * (gain_lin - 1)
        except Exception:
            pass
    return audio.astype(np.float32)


def apply_highpass(audio: np.ndarray) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    cutoff = random.uniform(60, 300)
    sos = butter(4, cutoff, btype='high', fs=SR, output='sos')
    return sosfilt(sos, audio).astype(np.float32)


def apply_lowpass(audio: np.ndarray) -> np.ndarray:
    from scipy.signal import butter, sosfilt
    cutoff = random.uniform(3000, 16000)
    sos = butter(4, cutoff, btype='low', fs=SR, output='sos')
    return sosfilt(sos, audio).astype(np.float32)


def apply_gain_jitter(audio: np.ndarray) -> np.ndarray:
    gain_db = random.uniform(-6, 6)
    return audio * (10 ** (gain_db / 20))


def apply_bitcrush(audio: np.ndarray) -> np.ndarray:
    bits = random.randint(8, 16)
    levels = 2 ** bits
    return (np.round(audio * levels) / levels).astype(np.float32)


# Registry of all degradation functions
DEGRADATIONS = {
    'noise':     lambda a: add_noise(a, random.choice(['white','pink','brown','hvac','hum']),
                                     random.uniform(5, 40)),
    'eq':        apply_eq,
    'highpass':  apply_highpass,
    'lowpass':   apply_lowpass,
    'gain':      apply_gain_jitter,
    'bitcrush':  apply_bitcrush,
}

def apply_random_degradations(audio: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    \"\"\"Apply a random subset of 3-6 degradations. Returns (degraded, list_of_names).\"\"\"
    n_augs = random.randint(AUGMENTATIONS_MIN, AUGMENTATIONS_MAX)
    chosen = random.sample(list(DEGRADATIONS.keys()), min(n_augs, len(DEGRADATIONS)))
    for name in chosen:
        audio = DEGRADATIONS[name](audio)
    # Hard clip to [-1, 1] as final safety
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32), chosen

print("[Degradation] Toolkit loaded ✓")
print(f"  Available: {list(DEGRADATIONS.keys())}")
print(f"  Per triple: {AUGMENTATIONS_MIN}-{AUGMENTATIONS_MAX} random degradations")
"""))

# ======================================================================
# CELL 10 — The Triple Engine
# ======================================================================
nb.cells.append(nbf.v4.new_code_cell("""\
# ═══════════════════════════════════════════════════════════════════════
# STAGE 4: THE TRIPLE ENGINE
# ═══════════════════════════════════════════════════════════════════════
# For each sterile vocal V:
#   1) Convolve with random bad IR (A)  → source_wet = V + A
#   2) Apply random degradations to source_wet
#   3) Convolve same V with random studio IR (C)  → target_wet = V + C
#   4) Get CLAP embedding of target_wet
#   5) Pack into batch
# ═══════════════════════════════════════════════════════════════════════

def convolve_and_trim(vocal: np.ndarray, ir: np.ndarray) -> np.ndarray:
    \"\"\"Convolve vocal with IR, trim to original length, peak-normalize.\"\"\"
    wet = fftconvolve(vocal, ir, mode='full')[:len(vocal)]
    peak = np.max(np.abs(wet))
    if peak > 1e-6:
        wet = wet / peak
    return wet.astype(np.float32)


def audio_to_int16(audio: np.ndarray) -> np.ndarray:
    \"\"\"Convert float32 [-1,1] to int16 for compact storage.\"\"\"
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16)


# ── Pre-shuffle vocal segments for reproducibility ──
random.shuffle(vocal_segments)

# ── Resume from checkpoint cursor ──
start_idx = ckpt.get('vocal_cursor', 0)
batch_id  = ckpt.get('batch_id', 0)
total     = ckpt.get('triples_completed', 0)

# ── Batch accumulators ──
batch_sources:  List[np.ndarray] = []
batch_targets:  List[np.ndarray] = []
batch_claps:    List[np.ndarray] = []
batch_meta:     List[dict]       = []

# ── Load CLAP cache ──
clap_cache_data = dict(np.load(OUTPUT / 'clap_cache.npz'))

print(f"\\n[Engine] Starting triple generation...")
print(f"  Vocal segments: {len(vocal_segments)}")
print(f"  Bad pool: {len(bad_pool)}, Studio pool: {len(studio_pool)}")
print(f"  Resuming from index {start_idx}, batch {batch_id}\\n")

t_start = time.time()
triples_this_run = 0
skipped = 0

for seg_idx in range(start_idx, len(vocal_segments)):
    # ── Budget check ──
    if get_output_size_gb() > MAX_OUTPUT_GB:
        print(f"\\n⚠  Output size limit reached ({MAX_OUTPUT_GB} GB). Stopping.")
        break

    seg = vocal_segments[seg_idx]
    V = seg['audio']

    # ── Pick random bad IR & studio IR ──
    bad_ir_id    = random.choice(bad_pool)
    studio_ir_id = random.choice(studio_pool)

    ir_A = all_irs[bad_ir_id]['audio']
    ir_C = all_irs[studio_ir_id]['audio']

    # ── Convolve ──
    source_wet = convolve_and_trim(V, ir_A)
    target_wet = convolve_and_trim(V, ir_C)

    # ── Random wet/dry mix on source (0.3-1.0) ──
    wet_dry = random.uniform(0.3, 1.0)
    source_wet = wet_dry * source_wet + (1 - wet_dry) * V

    # ── Degrade source only ──
    source_wet, aug_names = apply_random_degradations(source_wet)

    # ── QA: reject bad triples ──
    src_rms = np.sqrt(np.mean(source_wet ** 2))
    tgt_rms = np.sqrt(np.mean(target_wet ** 2))
    if src_rms < 1e-4 or tgt_rms < 1e-4:
        skipped += 1
        continue
    if np.any(np.isnan(source_wet)) or np.any(np.isnan(target_wet)):
        skipped += 1
        continue

    # ── CLAP embedding for target ──
    target_clap = clap_cache_data.get(studio_ir_id)
    if target_clap is None:
        # Fallback: compute live
        target_clap = get_clap_audio_embedding(target_wet)

    # ── Accumulate ──
    batch_sources.append(audio_to_int16(source_wet))
    batch_targets.append(audio_to_int16(target_wet))
    batch_claps.append(target_clap)
    batch_meta.append({
        'vocal_file':   seg['file'],
        'dataset':      seg['dataset'],
        'bad_ir':       bad_ir_id,
        'studio_ir':    studio_ir_id,
        'wet_dry':      round(wet_dry, 3),
        'degradations': aug_names,
    })

    # ── Flush batch when full ──
    if len(batch_sources) >= TRIPLES_PER_BATCH:
        batch_path = BATCH_DIR / f'batch_{batch_id:04d}.npz'
        np.savez(
            batch_path,
            source_audio  = np.stack(batch_sources),
            target_audio  = np.stack(batch_targets),
            target_clap   = np.stack(batch_claps),
        )
        # Save metadata separately (JSON, lightweight)
        meta_path = BATCH_DIR / f'batch_{batch_id:04d}_meta.json'
        with open(meta_path, 'w') as f:
            json.dump(batch_meta, f)

        triples_this_run += len(batch_sources)
        total += len(batch_sources)
        batch_id += 1

        # Checkpoint
        ckpt.update({
            'batch_id': batch_id,
            'triples_completed': total,
            'vocal_cursor': seg_idx + 1,
        })
        save_checkpoint(ckpt)

        elapsed = time.time() - t_start
        rate = triples_this_run / elapsed if elapsed > 0 else 0
        print(f"  Batch {batch_id-1:4d} saved | "
              f"triples: {total:,} total, {triples_this_run:,} this run | "
              f"{rate:.0f}/sec | "
              f"{get_output_size_gb():.1f} GB used")

        batch_sources.clear()
        batch_targets.clear()
        batch_claps.clear()
        batch_meta.clear()

# ── Flush remaining ──
if batch_sources:
    batch_path = BATCH_DIR / f'batch_{batch_id:04d}.npz'
    np.savez(
        batch_path,
        source_audio = np.stack(batch_sources),
        target_audio = np.stack(batch_targets),
        target_clap  = np.stack(batch_claps),
    )
    meta_path = BATCH_DIR / f'batch_{batch_id:04d}_meta.json'
    with open(meta_path, 'w') as f:
        json.dump(batch_meta, f)

    triples_this_run += len(batch_sources)
    total += len(batch_sources)
    batch_id += 1

    ckpt.update({
        'batch_id': batch_id,
        'triples_completed': total,
        'vocal_cursor': len(vocal_segments),
    })
    save_checkpoint(ckpt)

elapsed = time.time() - t_start
print(f"\\n[Engine] Run complete.")
print(f"  Triples this run: {triples_this_run:,}")
print(f"  Triples total:    {total:,}")
print(f"  Batches written:  {batch_id}")
print(f"  Skipped (QA):     {skipped}")
print(f"  Elapsed:          {elapsed/60:.1f} min")
print(f"  Output size:      {get_output_size_gb():.2f} GB")
"""))

# ======================================================================
# CELL 11 — Manifest & Summary
# ======================================================================
nb.cells.append(nbf.v4.new_code_cell("""\
# ═══════════════════════════════════════════════════════════════════════
# MANIFEST — checksums, statistics, verification
# ═══════════════════════════════════════════════════════════════════════

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

manifest = {
    'run_number':        ckpt['run_number'],
    'triples_total':     ckpt['triples_completed'],
    'batches':           ckpt['batch_id'],
    'sample_rate':       SR,
    'clip_seconds':      CLIP_SEC,
    'clip_samples':      CLIP_SAMPLES,
    'clap_dim':          CLAP_DIM,
    'bad_pool_size':     len(bad_pool),
    'studio_pool_size':  len(studio_pool),
    'output_size_gb':    round(get_output_size_gb(), 3),
    'batch_checksums':   {},
}

print("[Manifest] Computing checksums...")
for f in sorted(BATCH_DIR.glob('batch_*.npz')):
    manifest['batch_checksums'][f.name] = sha256_file(f)

with open(OUTPUT / 'manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"[Manifest] Saved ✓")
print(f"\\n{'='*60}")
print(f"  GENESIS DATA CURATION — RUN {ckpt['run_number']} COMPLETE")
print(f"{'='*60}")
print(f"  Total triples:  {manifest['triples_total']:,}")
print(f"  Batches:         {manifest['batches']}")
print(f"  Output size:     {manifest['output_size_gb']:.2f} GB")
print(f"  CLAP dim:        {manifest['clap_dim']}")
print(f"{'='*60}")

if ckpt['vocal_cursor'] < len(vocal_segments):
    remaining = len(vocal_segments) - ckpt['vocal_cursor']
    print(f"\\n⚠  {remaining} vocal segments remaining.")
    print(f"  To continue: save this output as a dataset,")
    print(f"  attach it to a new notebook as '{PREV_RUN_PATH.name}',")
    print(f"  and re-run this notebook.")
else:
    print(f"\\n✓  All vocal segments processed. Dataset complete.")
"""))

# ======================================================================
# CELL 12 — Chaining instructions
# ======================================================================
nb.cells.append(nbf.v4.new_markdown_cell("""\
## 🔗 Checkpoint Chaining (20 GB Limit)

If the output hit the size limit before processing all vocals:

1. **Save this notebook's output** as a Kaggle dataset (e.g. `genesis-data-run1`)
2. **Create a new notebook** (or re-run this one) and attach:
   - All the same input datasets (IRs, LJSpeech, VCTK, Language Identifier)
   - The previous output as input (update `PREV_RUN_PATH` in Cell 3)
3. **Run all cells** — the checkpoint system automatically skips completed work

Each run produces ~19 GB of training triples. Chain as many times as needed.

### Using the Data in Training

```python
# In the training notebook, load all batches from all runs:
import numpy as np
from pathlib import Path

run_dirs = [
    Path('/kaggle/input/genesis-data-run1/batches'),
    Path('/kaggle/input/genesis-data-run2/batches'),
    # ... add more runs
]

for run_dir in run_dirs:
    for batch_file in sorted(run_dir.glob('batch_*.npz')):
        data = np.load(batch_file)
        source_audio = data['source_audio']   # (N, 240000) int16
        target_audio = data['target_audio']   # (N, 240000) int16
        target_clap  = data['target_clap']    # (N, CLAP_DIM) float32
        # Convert int16 back to float32: audio = source_audio.astype(np.float32) / 32767
        # Compute STFT on-the-fly during training for memory efficiency
```
"""))

# ======================================================================
# Write the notebook
# ======================================================================
out_path = Path(__file__).parent / "01_acquire_and_augment.ipynb"
with open(out_path, "w") as f:
    nbf.write(nb, f)

print(f"✓ Notebook written to {out_path}")
print(f"  Cells: {len(nb.cells)} ({sum(1 for c in nb.cells if c.cell_type=='code')} code, "
      f"{sum(1 for c in nb.cells if c.cell_type=='markdown')} markdown)")
