# ══════════════════════════════════════════════════════════════════════════════
# Bark Scale Utilities (from Phase 1)
# ══════════════════════════════════════════════════════════════════════════════

def hz_to_bark(hz: float) -> float:
    return 13.0 * np.arctan(0.00076 * hz) + 3.5 * np.arctan((hz / 7500.0) ** 2)


def bark_to_hz(bark: float) -> float:
    return 600.0 * np.sinh(bark / 7.0)


def create_bark_filterbank(
    n_bands: int = BARK_N_BANDS,
    low_hz: float = BARK_LOW_HZ,
    high_hz: float = BARK_HIGH_HZ,
    n_fft: int = FFT_SIZE,
    sr: int = SR,
) -> np.ndarray:
    bark_low = hz_to_bark(low_hz)
    bark_high = hz_to_bark(high_hz)
    bark_centers = np.linspace(bark_low, bark_high, n_bands)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    filterbank = np.zeros((n_bands, len(freqs)), dtype=np.float64)
    for i in range(n_bands):
        center = bark_centers[i]
        left_edge = bark_low if i == 0 else (bark_centers[i - 1] + center) / 2.0
        right_edge = bark_high if i == n_bands - 1 else (center + bark_centers[i + 1]) / 2.0

        center_hz = bark_to_hz(center)
        left_hz = bark_to_hz(left_edge)
        right_hz = bark_to_hz(right_edge)

        for j, f in enumerate(freqs):
            if f < left_hz:
                filterbank[i, j] = 0.0
            elif left_hz <= f < center_hz and center_hz > left_hz:
                filterbank[i, j] = (f - left_hz) / (center_hz - left_hz)
            elif center_hz <= f < right_hz and right_hz > center_hz:
                filterbank[i, j] = (right_hz - f) / (right_hz - center_hz)
            elif f >= right_hz:
                filterbank[i, j] = 0.0
            else:
                filterbank[i, j] = 0.0

    return filterbank


_BARK_FILTERBANK = create_bark_filterbank()


# ══════════════════════════════════════════════════════════════════════════════
# LUFS (ITU-R BS.1770-4) (from Phase 1)
# ══════════════════════════════════════════════════════════════════════════════

def _design_k_weighting_filters(sr: int = SR):
    from scipy.signal import butter
    sos_pre = butter(1, 1500.0, btype="high", fs=sr, output="sos")
    f0 = 4000.0
    gain_db = 3.0
    gain_lin = 10.0 ** (gain_db / 20.0)
    sos_rlb = butter(2, f0, btype="high", fs=sr, output="sos")
    return sos_pre, sos_rlb


def compute_lufs_1d(audio: np.ndarray, sr: int = SR) -> float:
    from scipy.signal import sosfilt

    if len(audio) < sr * 0.1:
        filtered = sosfilt(_K_WEIGHTING_SOS, audio)
        mean_sq = np.mean(filtered ** 2)
        lufs = -0.691 + 10.0 * np.log10(mean_sq + 1e-20)
        return float(lufs)

    filtered = audio.copy()
    for sos in _K_WEIGHTING_SOS:
        filtered = sosfilt(sos, filtered)

    block_size = int(0.4 * sr)
    hop_size = int(0.1 * sr)
    n_blocks = max(1, (len(filtered) - block_size) // hop_size + 1)

    block_powers = np.zeros(n_blocks)
    for i in range(n_blocks):
        start = i * hop_size
        end = start + block_size
        if end > len(filtered):
            break
        block = filtered[start:end]
        block_powers[i] = np.mean(block ** 2)

    abs_gate = 10.0 ** ((-70.0 + 0.691) / 10.0)
    gated_mask = block_powers >= abs_gate

    if not np.any(gated_mask):
        return -70.0

    abs_gated_mean = np.mean(block_powers[gated_mask])
    rel_threshold = abs_gated_mean * (10.0 ** (-10.0 / 10.0))
    rel_gated_mask = block_powers >= rel_threshold

    if not np.any(rel_gated_mask):
        return -70.0

    final_mean = np.mean(block_powers[rel_gated_mask])
    lufs = -0.691 + 10.0 * np.log10(final_mean + 1e-20)
    return float(lufs)


_K_WEIGHTING_SOS = _design_k_weighting_filters(SR)


# ══════════════════════════════════════════════════════════════════════════════
# Crest Factor & ZCR (from Phase 1)
# ══════════════════════════════════════════════════════════════════════════════

def compute_crest_factor(audio: np.ndarray) -> float:
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    return 20.0 * np.log10(peak / (rms + 1e-10))


def compute_zcr(audio: np.ndarray) -> float:
    signs = np.sign(audio)
    signs[signs == 0] = 1
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings / (2.0 * (len(audio) - 1)))


# ══════════════════════════════════════════════════════════════════════════════
# 67D Metrics Extractor (from Phase 1)
# ══════════════════════════════════════════════════════════════════════════════

def compute_ltas_64d(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    from scipy.signal import stft

    f, t, Zxx = stft(audio, fs=sr, nperseg=FFT_SIZE, noverlap=FFT_SIZE - HOP_SIZE,
                     window="hann", return_onesided=True)
    power = np.abs(Zxx) ** 2

    band_energy = _BARK_FILTERBANK @ power
    mean_energy = np.mean(band_energy, axis=1)
    ltas = 10.0 * np.log10(mean_energy + 1e-10)
    return ltas.astype(np.float64)


def extract_metrics_67d(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    ltas = compute_ltas_64d(audio, sr)
    lufs = np.array([compute_lufs_1d(audio, sr)])
    crest = np.array([compute_crest_factor(audio)])
    zcr = np.array([compute_zcr(audio)])
    return np.concatenate([ltas, lufs, crest, zcr])


# ══════════════════════════════════════════════════════════════════════════════
# Clip Discovery (Pass 1)
# ══════════════════════════════════════════════════════════════════════════════

def discoverClips(pristine_paths, clusters, vctk_max=VCTK_MAX_CLIPS_PER_SPEAKER,
                  ljspeech_max=LJSpeech_MAX_CLIPS, daps_max=DAPS_MAX_CLIPS):
    """
    Scan datasets, pick clips per speaker with limits.
    Returns list of {"path": str, "speaker_id": str, "cluster_id": int}.
    """
    clips = []

    # VCTK: speakers starting with 'p'
    vctk_root = pristine_paths['vctk']
    if vctk_root.exists():
        for speaker_dir in sorted(vctk_root.iterdir()):
            if not speaker_dir.is_dir():
                continue
            speaker_id = speaker_dir.name
            if speaker_id not in clusters:
                continue

            wav_files = sorted(speaker_dir.glob("*.wav"))
            if not wav_files:
                continue

            # Sample up to vctk_max clips
            if len(wav_files) > vctk_max:
                wav_files = random.sample(wav_files, vctk_max)

            cluster_id = clusters[speaker_id]['cluster']
            for wav_path in wav_files:
                clips.append({
                    'path': str(wav_path),
                    'speaker_id': speaker_id,
                    'cluster_id': cluster_id,
                })

    # LJSpeech: single speaker LJ001
    ljspeech_root = pristine_paths['ljspeech']
    if ljspeech_root.exists():
        wav_files = sorted(ljspeech_root.glob("*.wav"))
        if wav_files and "LJ001" in clusters:
            if len(wav_files) > ljspeech_max:
                wav_files = random.sample(wav_files, ljspeech_max)

            cluster_id = clusters["LJ001"]['cluster']
            for wav_path in wav_files:
                clips.append({
                    'path': str(wav_path),
                    'speaker_id': 'LJ001',
                    'cluster_id': cluster_id,
                })

    # DAPS: speakers with f*/m* prefix
    daps_root = pristine_paths['daps']
    if daps_root.exists():
        daps_files = []
        for wav_file in sorted(daps_root.rglob("*.wav")):
            speaker_id = wav_file.stem.split("_")[0]
            if speaker_id in clusters:
                daps_files.append((wav_file, speaker_id))

        if daps_max and len(daps_files) > daps_max:
            daps_files = random.sample(daps_files, daps_max)

        for wav_path, speaker_id in daps_files:
            cluster_id = clusters[speaker_id]['cluster']
            clips.append({
                'path': str(wav_path),
                'speaker_id': speaker_id,
                'cluster_id': cluster_id,
            })

    return clips


# ══════════════════════════════════════════════════════════════════════════════
# JSONL Read/Write
# ══════════════════════════════════════════════════════════════════════════════

def writeChosenPristine(clips, output_path):
    """Write clip list to JSONL file."""
    with open(output_path, 'w') as f:
        for i, clip in enumerate(clips, 1):
            entry = {
                'pair_id': f'{i:08d}',
                'path': clip['path'],
                'speaker_id': clip['speaker_id'],
                'cluster_id': clip['cluster_id'],
            }
            f.write(json.dumps(entry) + '\n')


def readChosenPristine(jsonl_path):
    """Read clip list from JSONL file. Handles missing pair_id by generating from index."""
    entries = []
    with open(jsonl_path, 'r') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                entry = json.loads(line)
                if 'pair_id' not in entry:
                    entry['pair_id'] = f'{i:08d}'
                entries.append(entry)
    return entries


# ══════════════════════════════════════════════════════════════════════════════
# Random Parameter Generators
# ══════════════════════════════════════════════════════════════════════════════

def randomEqBands():
    n_bands = random.randint(1, 6)
    bands = []
    freqs_used = []

    for _ in range(n_bands):
        retries = 0
        while retries < 20:
            freq = np.exp(random.uniform(np.log(20), np.log(20000)))
            too_close = False
            for used_freq in freqs_used:
                if abs(freq - used_freq) / max(freq, used_freq) < 0.1:
                    too_close = True
                    break
            if not too_close:
                break
            retries += 1

        freqs_used.append(freq)
        band = {
            'freq_hz': round(float(freq), 1),
            'gain_db': round(random.uniform(-12, 12), 2),
            'q': round(random.uniform(0.1, 10), 2),
            'filter_type': random.choice(['peak', 'low_shelf', 'high_shelf']),
            'stereo_skew_db': 0.0,
            'dynamic_depth': 0.0,
        }
        bands.append(band)
    return bands


def randomGainParams():
    return {
        'gain_db': round(random.uniform(-12, 12), 2),
    }


def generateDegradationParams():
    return {
        'eq_bands': randomEqBands(),
        'gain': randomGainParams(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Degradation Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def applyDegradation(audio, sr, params):
    audio = audio.astype(np.float32)

    audio, _ = equalizer.process(audio, sr, bands=params['eq_bands'])
    audio, _ = gain1.process(audio, sr, **params['gain'])

    return audio


# ══════════════════════════════════════════════════════════════════════════════
# Audio Loading & Normalization
# ══════════════════════════════════════════════════════════════════════════════

def loadAndPrepareClip(audio_path, target_samples=CLIP_SAMPLES):
    """Load WAV, mono, resample to SR, pad/trim to target_samples."""
    audio, file_sr = sf.read(str(audio_path), dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != SR:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SR)

    if len(audio) < SR * 0.5:
        return None

    # Pad or trim to exact clip length
    if len(audio) < target_samples:
        padded = np.zeros(target_samples, dtype=np.float32)
        padded[:len(audio)] = audio
        audio = padded
    elif len(audio) > target_samples:
        # Take middle portion
        start = (len(audio) - target_samples) // 2
        audio = audio[start:start + target_samples]

    return audio


# ══════════════════════════════════════════════════════════════════════════════
# Save Functions
# ══════════════════════════════════════════════════════════════════════════════

def makeJsonSafe(obj):
    if isinstance(obj, dict):
        return {k: makeJsonSafe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [makeJsonSafe(v) for v in obj]
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def saveDegradedPair(pair_id, degraded, params, output_dir):
    """Save degraded WAV + params.json (no reference.wav)."""
    pair_dir = output_dir / 'pairs' / pair_id
    pair_dir.mkdir(parents=True, exist_ok=True)

    degraded_clip = np.clip(degraded, -1.0, 1.0)
    sf.write(str(pair_dir / f'{pair_id}.wav'),
             (degraded_clip * 32767).astype(np.int16), SR)

    params_clean = makeJsonSafe(params)
    with open(pair_dir / 'params.json', 'w') as f:
        json.dump(params_clean, f, indent=2)


def getOutputSizeGB(output_dir):
    total = sum(f.stat().st_size for f in output_dir.rglob('*') if f.is_file())
    return total / (1024 ** 3)


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint
# ══════════════════════════════════════════════════════════════════════════════

def saveCheckpoint(checkpoint_path, state):
    state['timestamp'] = time.time()
    with open(checkpoint_path, 'w') as f:
        json.dump(state, f, indent=2)


def loadCheckpoint(checkpoint_path):
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {'next_idx': 0, 'processed': 0, 'failed': 0}


def getCompletedPairIds(output_dir):
    """Return set of pair_ids that already have a WAV file."""
    pairs_dir = output_dir / 'pairs'
    if not pairs_dir.exists():
        return set()
    completed = set()
    for d in pairs_dir.iterdir():
        if d.is_dir():
            wav_files = list(d.glob('*.wav'))
            if wav_files:
                completed.add(d.name)
    return completed


# ══════════════════════════════════════════════════════════════════════════════
# Metadata
# ══════════════════════════════════════════════════════════════════════════════

def saveMetadata(clusters, identity_floors, n_pairs, clip_stats, output_dir):
    metadata = {
        'version': 2,
        'n_pairs': n_pairs,
        'sample_rate': SR,
        'clip_seconds': CLIP_SEC,
        'cluster_floors': identity_floors,
        'cluster_assignments': {sid: info['cluster'] for sid, info in clusters.items()},
        'n_speakers': len(clusters),
        'clip_stats': clip_stats,
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
