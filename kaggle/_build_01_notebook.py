#!/usr/bin/env python3
"""
Generates kaggle/01_acquire_and_augment.ipynb using nbformat.

Run once locally:
    python kaggle/_build_01_notebook.py

Spec reference: kaggle/01_docs/01.md
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
# CELL 1 — Title  (markdown)
# ======================================================================
nb.cells.append(nbf.v4.new_markdown_cell("""\
# 🧬 Genesis IR Head — Data Curation Pipeline

**Notebook 01**: Acquire → Sterilize → Augment → Target

This notebook produces training triples for Genesis's IR Synthesis Head:

| Tensor | Format | Description |
|--------|--------|-------------|
| `source_wet` | int16 waveform | Sterile vocal convolved with a **bad** IR + random degradations |
| `target_wet` | int16 waveform | Same sterile vocal convolved with a **target** MIT IR (ground truth) |
| `target_clap` | float32 vector | CLAP embedding of the target IR (pre-computed) |

**Checkpoint chaining**: Output is capped at ~19 GB per run. A `checkpoint.json`
tracks progress so the next run (using this output as input) continues where
we left off.

> **Pool definitions**
> * **Bad pool** — User-uploaded unlabeled IRs (uncontrolled environments)
> * **Target pool** — MIT Acoustical Reverberation Scene Statistics IRs (labeled, real-world rooms)
"""))

# ======================================================================
# CELL 2 — Installs  (code)
# ======================================================================
nb.cells.append(nbf.v4.new_code_cell("""\
!pip install -q noisereduce pyloudnorm soundfile librosa transformers torch
"""))

# ======================================================================
# CELL 3 — Phase 0: Environment & Path Configuration  (code)
# ======================================================================
nb.cells.append(nbf.v4.new_code_cell("""\
# ═══════════════════════════════════════════════════════════════════════
# PHASE 0 — Environment & Path Configuration
# ═══════════════════════════════════════════════════════════════════════

import os, json, hashlib, time, random, warnings, shutil, pickle
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
    'irs':       Path('/kaggle/input/datasets/itorousa/impulse-responses'),
    'mit_raw':   Path('/kaggle/input/datasets/kynthesis/mit-reverb-dataset'
                      '/MIT_Reverb_Dataset/MIT_Reverb_Dataset'),
    'ljspeech':  Path('/kaggle/input/datasets/dromosys/ljspeech'),
    'vctk':      Path('/kaggle/input/datasets/kynthesis/vctk-corpus'
                      '/VCTK-Corpus/wav48'),
    'langid_en': Path('/kaggle/input/datasets/shrivatssudhir'
                      '/language-identifier/english/clips'),
}

# Chaining — previous run's output (attached as input to this notebook).
# Adjust the path for each subsequent run.
PREV_RUN_PATH = Path('/kaggle/input/notebooks/itorousa/genesis-data-run1')

# ═══════════════════════════════════════════════════════════════════════
# OUTPUT
# ═══════════════════════════════════════════════════════════════════════
OUTPUT          = Path('/kaggle/working')
BATCH_DIR       = OUTPUT / 'batches'
CLAP_DIR        = OUTPUT / 'clap_model'
STERILIZED_DIR  = OUTPUT / 'sterilized_batches'
MIT_IR_DIR      = OUTPUT / 'irs' / 'mit_irs'

for d in [BATCH_DIR, CLAP_DIR, STERILIZED_DIR, MIT_IR_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════
# AUDIO PARAMETERS
# ═══════════════════════════════════════════════════════════════════════
SR            = 48_000
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
# CELL 4 — Phase 1: Checkpoint System  (code)
# ======================================================================
nb.cells.append(nbf.v4.new_code_cell("""\
# ═══════════════════════════════════════════════════════════════════════
# PHASE 1 — Aggressive Checkpoint Initialization & Hardware Protection
# ═══════════════════════════════════════════════════════════════════════
CHECKPOINT_PATH = OUTPUT / 'checkpoint.json'

def load_checkpoint() -> dict:
    \\"\\"\\"Load checkpoint: previous run → current working dir → fresh start.\\"\\"\\"
    # 1) Check previous run first (chaining)
    prev_ckpt = PREV_RUN_PATH / 'checkpoint.json'
    if prev_ckpt.exists():
        with open(prev_ckpt) as f:
            ckpt = json.load(f)
        ckpt['run_number'] += 1
        print(f"[Checkpoint] ♻ Resuming from previous run: "
              f"{ckpt['triples_completed']} triples done")
        return ckpt

    # 2) Check current working dir (kernel restart mid-session)
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)

    # 3) Fresh start
    return {
        'batch_id': 0,
        'triples_completed': 0,
        'vocal_cursor': 0,
        'run_number': 1,
    }

def save_checkpoint(ckpt: dict):
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump(ckpt, f, indent=2)

def get_output_size_gb() -> float:
    \\"\\"\\"Calculate the exact size of /kaggle/working in GB.\\"\\"\\"
    total = sum(f.stat().st_size for f in OUTPUT.rglob('*') if f.is_file())
    return total / (1024 ** 3)

ckpt = load_checkpoint()
print(f"[Checkpoint] Run #{ckpt['run_number']}, "
      f"{ckpt['triples_completed']} triples completed so far, "
      f"starting at vocal cursor {ckpt['vocal_cursor']}")
"""))

# ======================================================================
# CELL 5 — Phase 2: IR Acquisition & Pooling  (code)
# ======================================================================
nb.cells.append(nbf.v4.new_code_cell("""\
# ═══════════════════════════════════════════════════════════════════════
# PHASE 2 — Impulse Response Acquisition & Pooling
# ═══════════════════════════════════════════════════════════════════════
CATALOGUE_PATH = OUTPUT / 'ir_catalogue.json'

# ─── Fast path: reload from existing catalogue ───────────────────────
if CATALOGUE_PATH.exists():
    print("🟢 Found existing ir_catalogue.json — skipping audio-loading block.")
    with open(CATALOGUE_PATH, 'r') as f:
        ir_catalogue = json.load(f)

    all_irs: Dict[str, dict] = {}   # will NOT have audio yet; lazy-loaded later
    bad_pool:    List[str] = []
    target_pool: List[str] = []

    for ir_id, feats in ir_catalogue.items():
        if feats['source'] == 'mit':
            target_pool.append(ir_id)
        elif feats['rt60'] > 0.25 or feats['c50'] < 8:
            bad_pool.append(ir_id)
        else:
            bad_pool.append(ir_id)

    # Safety: ensure target pool is adequate
    if len(target_pool) < 20:
        mit_ids = [k for k, v in ir_catalogue.items() if v['source'] == 'mit']
        target_pool = mit_ids if mit_ids else list(ir_catalogue.keys())[:50]
        bad_pool = [k for k in ir_catalogue if k not in target_pool]

    print(f"🟢 Loaded {len(ir_catalogue)} IRs from catalogue")
    print(f"🟢 Bad pool: {len(bad_pool)} | Target pool: {len(target_pool)}")
    _catalogue_loaded_from_disk = True

else:
    _catalogue_loaded_from_disk = False

    # ─── Stage 1: Copy / acquire MIT IRs ─────────────────────────────
    prev_mit = PREV_RUN_PATH / 'irs' / 'mit_irs'
    if prev_mit.exists() and not any(MIT_IR_DIR.iterdir()):
        print("🟢 [Acquire] Copying MIT IRs from previous run...")
        shutil.copytree(prev_mit, MIT_IR_DIR, dirs_exist_ok=True)
        print("🟢 [Acquire] MIT IRs staged from previous run ✓")
    elif PATHS['mit_raw'].exists() and not any(MIT_IR_DIR.iterdir()):
        print("🟢 [Acquire] Copying MIT IRs from raw dataset...")
        shutil.copytree(PATHS['mit_raw'], MIT_IR_DIR, dirs_exist_ok=True)
        print("🟢 [Acquire] MIT IRs staged from dataset ✓")
    else:
        print("🟢 [Acquire] MIT IRs already staged or unavailable")

    # ─── Stage 2: Load & normalize every IR to 48 kHz mono ───────────
    def load_and_normalize_ir(filepath: Path, target_sr: int = SR) -> Optional[np.ndarray]:
        \\"\\"\\"Load an IR file, force mono / target SR, peak-normalize.\\"\\"\\"
        try:
            audio, _ = librosa.load(str(filepath), sr=target_sr, mono=True)
            if len(audio) < 64:
                return None
            peak = np.max(np.abs(audio))
            if peak > 1e-6:
                audio = audio / peak
            return audio.astype(np.float32)
        except Exception:
            return None

    print("\\n🟢 [Acquire] Loading all impulse responses...")
    all_irs: Dict[str, dict] = {}

    # Bad pool — user's unlabeled IRs
    for ext in ('*.irs', '*.wav'):
        for f in sorted(PATHS['irs'].rglob(ext)):
            ir = load_and_normalize_ir(f)
            if ir is not None:
                all_irs[f'user_{f.stem}'] = {'audio': ir, 'source': 'user'}

    # Target pool — MIT labeled IRs
    if MIT_IR_DIR.exists():
        for f in sorted(MIT_IR_DIR.rglob('*.wav')):
            ir = load_and_normalize_ir(f)
            if ir is not None:
                all_irs[f'mit_{f.stem}'] = {'audio': ir, 'source': 'mit'}

    print(f"🟢 [Acquire] Loaded {len(all_irs)} IRs "
          f"(user/bad: {sum(1 for v in all_irs.values() if v['source']=='user')}, "
          f"MIT/target: {sum(1 for v in all_irs.values() if v['source']=='mit')})")

    # ─── Stage 3: Classify & build pools ─────────────────────────────
    def compute_ir_features(ir_audio: np.ndarray, sr: int = SR) -> dict:
        \\"\\"\\"Compute RT60 estimate, spectral centroid, and clarity (C50).\\"\\"\\"
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

        # Clarity C50 — early-to-late energy ratio at 50 ms
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
    bad_pool:    List[str] = []
    target_pool: List[str] = []

    for ir_id, ir_data in all_irs.items():
        feats = compute_ir_features(ir_data['audio'])
        ir_data['features'] = feats
        ir_catalogue[ir_id] = {'source': ir_data['source'], **feats}

        # MIT IRs → target pool (labeled ground-truth rooms)
        # User IRs → bad pool (unlabeled/degraded)
        if ir_data['source'] == 'mit':
            target_pool.append(ir_id)
        else:
            bad_pool.append(ir_id)

    # Safety: ensure target pool is adequate
    if len(target_pool) < 20:
        mit_ids = sorted(
            [k for k, v in all_irs.items() if v['source'] == 'mit'],
            key=lambda k: all_irs[k]['features']['c50'], reverse=True
        )
        target_pool = mit_ids if mit_ids else list(all_irs.keys())[:50]
        bad_pool = [k for k in all_irs if k not in target_pool]

    print(f"🟢 [Classify] Bad pool:    {len(bad_pool)} IRs")
    print(f"🟢 [Classify] Target pool: {len(target_pool)} IRs")

    # 💾 Save catalogue
    with open(CATALOGUE_PATH, 'w') as f:
        json.dump(ir_catalogue, f, indent=2)
    print("🟢 [Classify] ir_catalogue.json saved ✓")
"""))

# ======================================================================
# CELL 6 — Phase 3: CLAP Target Embedding Cache  (code)
# ======================================================================
nb.cells.append(nbf.v4.new_code_cell("""\
import torch
from transformers import ClapModel, ClapProcessor

# ═══════════════════════════════════════════════════════════════════════
# PHASE 3 — CLAP Target Embedding Cache
# ═══════════════════════════════════════════════════════════════════════

CLAP_MODEL_ID = "laion/larger_clap_music_and_speech"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🟢 [CLAP] Device: {device}")

# ─── Model loading chain: local → previous run → download ────────────
prev_clap = PREV_RUN_PATH / 'clap_model'

if (CLAP_DIR / 'config.json').exists():
    print("🟢 [CLAP] Loading frozen model from current working dir...")
    clap_processor = ClapProcessor.from_pretrained(CLAP_DIR)
    clap_model = ClapModel.from_pretrained(CLAP_DIR).to(device).eval()

elif prev_clap.exists() and (prev_clap / 'config.json').exists():
    print("🟢 [CLAP] Copying frozen model from previous run...")
    shutil.copytree(prev_clap, CLAP_DIR, dirs_exist_ok=True)
    clap_processor = ClapProcessor.from_pretrained(CLAP_DIR)
    clap_model = ClapModel.from_pretrained(CLAP_DIR).to(device).eval()

else:
    print("🟢 [CLAP] Downloading model from Hugging Face...")
    clap_processor = ClapProcessor.from_pretrained(CLAP_MODEL_ID)
    clap_model = ClapModel.from_pretrained(CLAP_MODEL_ID).to(device).eval()
    clap_model.save_pretrained(CLAP_DIR)
    clap_processor.save_pretrained(CLAP_DIR)
    print(f"🟢 [CLAP] Frozen model saved to {CLAP_DIR}")

CLAP_DIM = clap_model.config.projection_dim
print(f"🟢 [CLAP] Loaded — embedding dim = {CLAP_DIM}")


def get_clap_audio_embedding(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    \\"\\"\\"
    Encode audio through CLAP's audio tower. Returns (CLAP_DIM,) float32.

    ⚠ Critical: extract .pooler_output from the wrapper before calling .cpu().
    \\"\\"\\"
    inputs = clap_processor(
        audios=audio, sampling_rate=sr, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = clap_model.get_audio_features(**inputs)
        emb = outputs.pooler_output
    return emb.cpu().numpy().flatten().astype(np.float32)


# ─── Pre-compute CLAP embeddings for every target-pool IR ─────────────
# Convolve each IR with 3s of white noise so CLAP has a rich scene.
CLAP_CACHE_PATH = OUTPUT / 'clap_cache.npz'
prev_cache = PREV_RUN_PATH / 'clap_cache.npz'

if CLAP_CACHE_PATH.exists():
    print("🟢 [CLAP] Found existing clap_cache.npz — skipping embedding compute.")
    clap_cache = dict(np.load(CLAP_CACHE_PATH))
    print(f"🟢 [CLAP] Loaded {len(clap_cache)} embeddings from cache ✓  (dim={CLAP_DIM})")

elif prev_cache.exists():
    print("🟢 [CLAP] Copying clap_cache.npz from previous run...")
    shutil.copy2(prev_cache, CLAP_CACHE_PATH)
    clap_cache = dict(np.load(CLAP_CACHE_PATH))
    print(f"🟢 [CLAP] Loaded {len(clap_cache)} embeddings from previous run ✓")

else:
    print(f"\\n🟢 [CLAP] Pre-computing embeddings for {len(target_pool)} target IRs...")

    # Need IR audio loaded — if catalogue was loaded from disk, lazy-load now
    if _catalogue_loaded_from_disk:
        print("   (Lazy-loading IR audio for embedding computation...)")
        def _lazy_load_ir(ir_id: str) -> Optional[np.ndarray]:
            if ir_id.startswith('mit_'):
                for f in MIT_IR_DIR.rglob('*.wav'):
                    if f.stem == ir_id[4:]:
                        audio, _ = librosa.load(str(f), sr=SR, mono=True)
                        peak = np.max(np.abs(audio))
                        return (audio / peak).astype(np.float32) if peak > 1e-6 else audio
            elif ir_id.startswith('user_'):
                for ext in ('*.irs', '*.wav'):
                    for f in PATHS['irs'].rglob(ext):
                        if f.stem == ir_id[5:]:
                            audio, _ = librosa.load(str(f), sr=SR, mono=True)
                            peak = np.max(np.abs(audio))
                            return (audio / peak).astype(np.float32) if peak > 1e-6 else audio
            return None

        all_irs = {}
        for ir_id in target_pool + bad_pool:
            a = _lazy_load_ir(ir_id)
            if a is not None:
                all_irs[ir_id] = {'audio': a, 'source': ir_catalogue[ir_id]['source']}

    ref_noise = np.random.randn(SR * 3).astype(np.float32) * 0.1

    clap_cache: Dict[str, np.ndarray] = {}
    for i, ir_id in enumerate(target_pool):
        if ir_id not in all_irs:
            continue
        ir_audio = all_irs[ir_id]['audio']
        scene = fftconvolve(ref_noise, ir_audio, mode='full')[:SR * 3]
        scene = scene / (np.max(np.abs(scene)) + 1e-8)
        clap_cache[ir_id] = get_clap_audio_embedding(scene)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(target_pool)}")

    np.savez(CLAP_CACHE_PATH, **clap_cache)
    print(f"🟢 [CLAP] Cached {len(clap_cache)} embeddings ✓  (dim={CLAP_DIM})")
"""))

# ======================================================================
# CELL 7 — Phase 4: Vocal Sterilization  (code)
# ======================================================================
nb.cells.append(nbf.v4.new_code_cell("""\
import pyloudnorm as pyln

# ═══════════════════════════════════════════════════════════════════════
# PHASE 4 — Vocal Sterilization & "Dead" Audio Guarantee
# ═══════════════════════════════════════════════════════════════════════

meter = pyln.Meter(SR)
STATE_FILE = STERILIZED_DIR / 'sterilize_state.json'

# ─── Check if a previous run already completed sterilization ──────────
prev_sterilized = PREV_RUN_PATH / 'sterilized_batches'
prev_state_file = prev_sterilized / 'sterilize_state.json' if prev_sterilized.exists() else None

_skip_sterilization = False
if prev_state_file and prev_state_file.exists():
    with open(prev_state_file) as f:
        prev_state = json.load(f)
    if prev_state.get('completed', False):
        print("🟢 [Sterilize] Previous run completed sterilization. Copying batches...")
        shutil.copytree(prev_sterilized, STERILIZED_DIR, dirs_exist_ok=True)
        _skip_sterilization = True
        print("🟢 [Sterilize] Sterilized batches copied from previous run ✓")

if STATE_FILE.exists() and not _skip_sterilization:
    with open(STATE_FILE) as f:
        _st = json.load(f)
    if _st.get('completed', False):
        print("🟢 [Sterilize] Already completed in this working dir. Skipping.")
        _skip_sterilization = True

if not _skip_sterilization:
    def discover_audio_files() -> List[Tuple[Path, str]]:
        \\"\\"\\"Collect all vocal files from all datasets.\\"\\"\\"
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
        \\"\\"\\"Load → noise-reduce → trim → LUFS normalize → segment into 5s windows.\\"\\"\\"
        try:
            audio, _ = librosa.load(str(filepath), sr=SR, mono=True)
        except Exception:
            return []

        if len(audio) < SR * 1.0:
            return []

        # Spectral noise reduction — strip residual room tone / hiss
        audio = nr.reduce_noise(y=audio, sr=SR, stationary=True, prop_decrease=0.85)

        # Trim absolute silence
        audio, _ = librosa.effects.trim(audio, top_db=40)
        if len(audio) < SR * 1.5:
            return []

        # Normalize loudness to -23 LUFS
        try:
            loudness = meter.integrated_loudness(audio)
            if loudness > -70:
                audio = pyln.normalize.loudness(audio, loudness, -23.0)
        except Exception:
            pass

        # Segment into rigid CLIP_SAMPLES (5.0s) chunks
        segments = []
        for start in range(0, len(audio) - SR, CLIP_SAMPLES):
            chunk = audio[start : start + CLIP_SAMPLES]
            if len(chunk) < CLIP_SAMPLES:
                chunk = np.pad(chunk, (0, CLIP_SAMPLES - len(chunk)))

            rms = np.sqrt(np.mean(chunk ** 2))
            if rms < 1e-4:
                continue

            segments.append({
                'audio':   chunk.astype(np.float32),
                'file':    filepath.name,
                'dataset': tag,
            })
        return segments

    # ── Discover & deterministic shuffle ──
    print("🟢 [Sterilize] Discovering audio files...")
    all_audio_files = discover_audio_files()

    # Sort alphabetically → shuffle with fixed seed for reproducibility
    all_audio_files.sort(key=lambda x: str(x[0]))
    random.Random(42).shuffle(all_audio_files)
    print(f"🟢 [Sterilize] Found {len(all_audio_files)} source files")

    # ── Resume from sterilize checkpoint ──
    cursor_start = 0
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            cursor_start = json.load(f).get('cursor', 0)

    STERILIZE_CHUNK = 500
    vocal_segments: List[dict] = []

    print(f"🟢 [Sterilize] Processing from file index {cursor_start}...")
    for i in range(cursor_start, len(all_audio_files)):
        fpath, tag = all_audio_files[i]

        # Budget check
        if get_output_size_gb() > MAX_OUTPUT_GB:
            print(f"\\n⚠ Output size limit reached. Saving sterilization progress.")
            break

        if i % 200 == 0 and i > cursor_start:
            print(f"🟢 Processed {i}/{len(all_audio_files)} files → "
                  f"{len(vocal_segments)} segments in RAM buffer")

        segs = sterilize_and_segment(fpath, tag)
        vocal_segments.extend(segs)

        # 💾 Granular disk flushing every STERILIZE_CHUNK files
        if (i + 1) % STERILIZE_CHUNK == 0 or (i + 1) == len(all_audio_files):
            batch_index = (i + 1) // STERILIZE_CHUNK
            batch_path = STERILIZED_DIR / f"sterilized_batch_{batch_index:04d}.pkl"

            with open(batch_path, 'wb') as f:
                pickle.dump(vocal_segments, f)

            completed = (i + 1) >= len(all_audio_files)
            with open(STATE_FILE, 'w') as f:
                json.dump({'cursor': i + 1, 'completed': completed}, f)

            print(f"💾 Saved {len(vocal_segments)} segments to {batch_path.name}. RAM cleared.")
            vocal_segments.clear()

# ── Aggregate: read all .pkl batches to compile the segment count ──
print("\\n🟢 [Sterilize] Compiling dataset breakdown from disk...")
ds_counts = defaultdict(int)
total_segments = 0

for batch_file in sorted(STERILIZED_DIR.glob('*.pkl')):
    with open(batch_file, 'rb') as f:
        batch_data = pickle.load(f)
        total_segments += len(batch_data)
        for s in batch_data:
            ds_counts[s['dataset'].split('_')[0]] += 1

print(f"🟢 [Sterilize] Total sterile segments: {total_segments}")
for ds, cnt in sorted(ds_counts.items()):
    print(f"  {ds}: {cnt}")
"""))

# ======================================================================
# CELL 8 — Phase skip notice (markdown)
# ======================================================================
nb.cells.append(nbf.v4.new_markdown_cell("""\
---
> ⚠️ **Phase Skip**: Phases 2–4 above can be skipped entirely if a previous run
> completed them successfully. The notebook detects existing `ir_catalogue.json`,
> `clap_cache.npz`, and `sterilize_state.json` to bypass redundant computation.
> Previous run files are located at `/kaggle/input/notebooks/itorousa/genesis-data-run#`.
---
"""))

# ======================================================================
# CELL 9 — Phase 5: Messy DSP Toolkit  (code)
# ======================================================================
nb.cells.append(nbf.v4.new_code_cell("""\
# ═══════════════════════════════════════════════════════════════════════
# PHASE 5 — The Messy DSP Toolkit
# ═══════════════════════════════════════════════════════════════════════
# Random, destructive audio effects applied ONLY to the source_wet input
# to simulate terrible recording environments.

def add_noise(audio: np.ndarray, noise_type: str, snr_db: float) -> np.ndarray:
    \\"\\"\\"Mix coloured noise at the target SNR (5–40 dB).\\"\\"\\"
    n = len(audio)
    if noise_type == 'white':
        noise = np.random.randn(n)
    elif noise_type == 'pink':
        freqs = np.fft.rfftfreq(n, 1 / SR)
        freqs[0] = 1
        S = 1.0 / np.sqrt(freqs)
        noise = np.fft.irfft(S * np.exp(2j * np.pi * np.random.rand(len(S))))[:n]
    elif noise_type == 'brown':
        noise = np.cumsum(np.random.randn(n))
        noise -= np.mean(noise)
    elif noise_type == 'hvac':
        noise = np.random.randn(n)
        from scipy.signal import butter, sosfilt
        sos = butter(4, [100, 1000], btype='band', fs=SR, output='sos')
        noise = sosfilt(sos, noise)
    elif noise_type == 'hum':
        t = np.arange(n) / SR
        base_freq = random.choice([50, 60])
        noise = np.zeros(n)
        for h in range(1, 6):
            amp = 1.0 / h
            noise += amp * np.sin(2 * np.pi * base_freq * h * t
                                  + random.uniform(0, 2 * np.pi))
    else:
        noise = np.random.randn(n)

    sig_power = np.mean(audio ** 2) + 1e-12
    noise_power = np.mean(noise ** 2) + 1e-12
    target_noise_power = sig_power / (10 ** (snr_db / 10))
    noise = noise * np.sqrt(target_noise_power / noise_power)
    return audio + noise.astype(np.float32)


def apply_eq(audio: np.ndarray) -> np.ndarray:
    \\"\\"\\"Random 3-band parametric EQ (simulates mic coloration).\\"\\"\\"
    from scipy.signal import butter, sosfilt
    bands = [(80, 300), (300, 3000), (3000, 12000)]
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
    \\"\\"\\"Bitcrushing / hard clipping — quantize audio resolution.\\"\\"\\"
    bits = random.randint(8, 16)
    levels = 2 ** bits
    return (np.round(audio * levels) / levels).astype(np.float32)


def apply_hard_clip(audio: np.ndarray) -> np.ndarray:
    \\"\\"\\"Harsh mathematical clipping at a random threshold.\\"\\"\\"
    threshold = random.uniform(0.3, 0.9)
    return np.clip(audio, -threshold, threshold).astype(np.float32)


# Registry of all degradation functions
DEGRADATIONS = {
    'noise':     lambda a: add_noise(a, random.choice(
                     ['white', 'pink', 'brown', 'hvac', 'hum']),
                     random.uniform(5, 40)),
    'eq':        apply_eq,
    'highpass':  apply_highpass,
    'lowpass':   apply_lowpass,
    'gain':      apply_gain_jitter,
    'bitcrush':  apply_bitcrush,
    'clip':      apply_hard_clip,
}

def apply_random_degradations(audio: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    \\"\\"\\"Apply a random subset of 3–6 degradations. Returns (degraded, names).\\"\\"\\"
    n_augs = random.randint(AUGMENTATIONS_MIN, AUGMENTATIONS_MAX)
    chosen = random.sample(list(DEGRADATIONS.keys()), min(n_augs, len(DEGRADATIONS)))
    for name in chosen:
        audio = DEGRADATIONS[name](audio)
    audio = np.clip(audio, -1.0, 1.0)
    return audio.astype(np.float32), chosen

print("🟢 [Degradation] Toolkit loaded ✓")
print(f"  Available: {list(DEGRADATIONS.keys())}")
print(f"  Per triple: {AUGMENTATIONS_MIN}–{AUGMENTATIONS_MAX} random degradations")
"""))

# ======================================================================
# CELL 10 — Phase 6: Triple Engine  (code)
# ======================================================================
nb.cells.append(nbf.v4.new_code_cell("""\
# ═══════════════════════════════════════════════════════════════════════
# PHASE 6 — The Training Data Engine (Triple Generation)
# ═══════════════════════════════════════════════════════════════════════
# For each dead dry vocal V:
#   1) Pick one random Bad IR (A) and one random Target MIT IR (C)
#   2) target_wet  = V ⊛ C  (ground truth)
#   3) source_wet  = (V ⊛ A) × wet_ratio + V × (1 - wet_ratio)
#   4) Degrade source_wet through DSP toolkit
#   5) Fetch CLAP embedding for C
#   6) QA → pack into batch
# ═══════════════════════════════════════════════════════════════════════

def convolve_and_trim(vocal: np.ndarray, ir: np.ndarray) -> np.ndarray:
    \\"\\"\\"Convolve vocal with IR, trim to exactly 5s, peak-normalize.\\"\\"\\"
    wet = fftconvolve(vocal, ir, mode='full')[:CLIP_SAMPLES]
    peak = np.max(np.abs(wet))
    if peak > 1e-6:
        wet = wet / peak
    return wet.astype(np.float32)


def audio_to_int16(audio: np.ndarray) -> np.ndarray:
    \\"\\"\\"Convert float32 [-1,1] to int16 for compact storage.\\"\\"\\"
    return (np.clip(audio, -1, 1) * 32767).astype(np.int16)


# ── Load sterilized vocal segments from disk ──
print("🟢 [Engine] Loading sterilized vocal segments from disk...")
vocal_segments: List[dict] = []
for pkl_file in sorted(STERILIZED_DIR.glob('sterilized_batch_*.pkl')):
    with open(pkl_file, 'rb') as f:
        vocal_segments.extend(pickle.load(f))
print(f"🟢 [Engine] Loaded {len(vocal_segments)} vocal segments")

# ── Ensure IR audio is loaded (lazy-load if catalogue was from disk) ──
if not all_irs or not any('audio' in v for v in all_irs.values() if isinstance(v, dict)):
    print("🟢 [Engine] Lazy-loading IR audio for convolution...")
    def _load_ir(ir_id: str) -> Optional[np.ndarray]:
        if ir_id.startswith('mit_'):
            for f in MIT_IR_DIR.rglob('*.wav'):
                if f.stem == ir_id[4:]:
                    audio, _ = librosa.load(str(f), sr=SR, mono=True)
                    peak = np.max(np.abs(audio))
                    return (audio / peak).astype(np.float32) if peak > 1e-6 else audio
        elif ir_id.startswith('user_'):
            for ext in ('*.irs', '*.wav'):
                for f in PATHS['irs'].rglob(ext):
                    if f.stem == ir_id[5:]:
                        audio, _ = librosa.load(str(f), sr=SR, mono=True)
                        peak = np.max(np.abs(audio))
                        return (audio / peak).astype(np.float32) if peak > 1e-6 else audio
        return None

    all_irs = {}
    for ir_id in target_pool + bad_pool:
        a = _load_ir(ir_id)
        if a is not None:
            all_irs[ir_id] = {'audio': a, 'source': ir_catalogue.get(ir_id, {}).get('source', 'unknown')}
    print(f"🟢 [Engine] Loaded {len(all_irs)} IRs")

# ── Shuffle segments for reproducibility ──
random.shuffle(vocal_segments)

# ── Resume from checkpoint ──
start_idx = ckpt.get('vocal_cursor', 0)
batch_id  = ckpt.get('batch_id', 0)
total     = ckpt.get('triples_completed', 0)

# ── Batch accumulators ──
batch_sources:  List[np.ndarray] = []
batch_targets:  List[np.ndarray] = []
batch_claps:    List[np.ndarray] = []
batch_meta:     List[dict]       = []

# ── Load CLAP cache ──
clap_cache_data = dict(np.load(CLAP_CACHE_PATH))

print(f"\\n🟢 [Engine] Starting triple generation...")
print(f"  Vocal segments: {len(vocal_segments)}")
print(f"  Bad pool: {len(bad_pool)}, Target pool: {len(target_pool)}")
print(f"  Resuming from index {start_idx}, batch {batch_id}\\n")

t_start = time.time()
triples_this_run = 0
skipped = 0

for seg_idx in range(start_idx, len(vocal_segments)):
    # ── Budget check ──
    if seg_idx % 100 == 0 and get_output_size_gb() > MAX_OUTPUT_GB:
        print(f"\\n⚠ Output size limit reached ({MAX_OUTPUT_GB} GB). Stopping.")
        break

    seg = vocal_segments[seg_idx]
    V = seg['audio']

    # ── Pick random bad IR & target IR ──
    bad_ir_id    = random.choice(bad_pool)
    target_ir_id = random.choice(target_pool)

    if bad_ir_id not in all_irs or target_ir_id not in all_irs:
        skipped += 1
        continue

    ir_A = all_irs[bad_ir_id]['audio']
    ir_C = all_irs[target_ir_id]['audio']

    # ── Create target audio (ground truth): V ⊛ target_IR ──
    target_wet = convolve_and_trim(V, ir_C)

    # ── Create messy source: (V ⊛ bad_IR) mixed with raw V ──
    source_wet = convolve_and_trim(V, ir_A)
    wet_dry = random.uniform(0.3, 1.0)
    source_wet = wet_dry * source_wet + (1 - wet_dry) * V[:len(source_wet)]

    # ── Degrade source only ──
    source_wet, aug_names = apply_random_degradations(source_wet)

    # ── QA: reject silent / NaN triples ──
    src_rms = np.sqrt(np.mean(source_wet ** 2))
    tgt_rms = np.sqrt(np.mean(target_wet ** 2))
    if src_rms < 1e-4 or tgt_rms < 1e-4:
        skipped += 1
        continue
    if np.any(np.isnan(source_wet)) or np.any(np.isnan(target_wet)):
        skipped += 1
        continue

    # ── CLAP embedding for target IR ──
    target_clap = clap_cache_data.get(target_ir_id)
    if target_clap is None:
        target_clap = get_clap_audio_embedding(target_wet)

    # ── Quantize & accumulate ──
    batch_sources.append(audio_to_int16(source_wet))
    batch_targets.append(audio_to_int16(target_wet))
    batch_claps.append(target_clap)
    batch_meta.append({
        'vocal_file':   seg['file'],
        'dataset':      seg['dataset'],
        'bad_ir':       bad_ir_id,
        'target_ir':    target_ir_id,
        'wet_dry':      round(wet_dry, 3),
        'degradations': aug_names,
    })

    # ── 💾 Flush batch when full ──
    if len(batch_sources) >= TRIPLES_PER_BATCH:
        batch_path = BATCH_DIR / f'batch_{batch_id:04d}.npz'
        np.savez(
            batch_path,
            source_audio  = np.stack(batch_sources),
            target_audio  = np.stack(batch_targets),
            target_clap   = np.stack(batch_claps),
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
"""))

# ======================================================================
# CELL 11 — Phase skip notice #2 (markdown)
# ======================================================================
nb.cells.append(nbf.v4.new_markdown_cell("""\
---
> ⚠️ **Phase Skip**: Phases 5–6 above can be skipped if a previous run already
> generated sufficient training batches. Attach the previous output as input and
> the checkpoint system will resume from where it left off.
---
"""))

# ======================================================================
# CELL 12 — Phase 7: Manifest & Pipeline Conclusion  (code)
# ======================================================================
nb.cells.append(nbf.v4.new_code_cell("""\
# ═══════════════════════════════════════════════════════════════════════
# PHASE 7 — Manifest & Pipeline Conclusion
# ═══════════════════════════════════════════════════════════════════════

# ── Flush any remaining triples (<500) ──
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

# ── SHA-256 checksums ──
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
    'target_pool_size':  len(target_pool),
    'output_size_gb':    round(get_output_size_gb(), 3),
    'batch_checksums':   {},
}

print("[Manifest] Computing SHA-256 checksums...")
for f in sorted(BATCH_DIR.glob('batch_*.npz')):
    manifest['batch_checksums'][f.name] = sha256_file(f)

with open(OUTPUT / 'manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print(f"[Manifest] Saved ✓")
print(f"\\n{'='*60}")
print(f"  GENESIS DATA CURATION — RUN {ckpt['run_number']} COMPLETE")
print(f"{'='*60}")
print(f"  Total triples:   {manifest['triples_total']:,}")
print(f"  Batches:          {manifest['batches']}")
print(f"  Output size:      {manifest['output_size_gb']:.2f} GB")
print(f"  CLAP dim:         {manifest['clap_dim']}")
print(f"  Elapsed:          {elapsed/60:.1f} min")
print(f"{'='*60}")

if ckpt['vocal_cursor'] < len(vocal_segments):
    remaining = len(vocal_segments) - ckpt['vocal_cursor']
    print(f"\\n⚠  {remaining} vocal segments remaining.")
    print(f"  To continue:")
    print(f"    1. Save this notebook's output as a Kaggle dataset")
    print(f"    2. Update PREV_RUN_PATH in Cell 3 to point to it")
    print(f"    3. Create a new notebook, attach the same input datasets")
    print(f"    4. Run all cells — the checkpoint system will resume")
else:
    print(f"\\n🟢 All vocal segments processed. Dataset complete!")
    print(f"   Output is ready for Genesis's STFT Dataloader in the training phase.")
"""))

# ======================================================================
# CELL 13 — Chaining Instructions  (markdown)
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
# Write the notebook to disk
# ======================================================================
out_path = Path(__file__).parent / "01_acquire_and_augment.ipynb"
with open(out_path, "w") as f:
    nbf.write(nb, f)

print(f"✓ Notebook written to {out_path}")
print(f"  Cells: {len(nb.cells)} ({sum(1 for c in nb.cells if c.cell_type=='code')} code, "
      f"{sum(1 for c in nb.cells if c.cell_type=='markdown')} markdown)")
