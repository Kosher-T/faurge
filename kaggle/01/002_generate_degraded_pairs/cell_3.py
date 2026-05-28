# ══════════════════════════════════════════════════════════════════════════════
# Cache Functions
# ══════════════════════════════════════════════════════════════════════════════

def savePairList(pairs, src_counts, ref_counts, cache_path):
    """Save pair list and balance counts to JSON for deterministic re-runs."""
    data = {
        'pairs': pairs,
        'source_counts': dict(src_counts),
        'ref_counts': dict(ref_counts),
    }
    with open(cache_path, 'w') as f:
        json.dump(data, f, indent=2)


def loadPairList(cache_path):
    """Load cached pair list and balance counts."""
    with open(cache_path, 'r') as f:
        data = json.load(f)
    return data['pairs'], defaultdict(int, data['source_counts']), defaultdict(int, data['ref_counts'])


def pairExists(pair_id, output_dir):
    """Check if a pair has already been processed (has degraded.wav)."""
    pair_dir = output_dir / 'pairs' / f'{pair_id:08d}'
    return (pair_dir / 'degraded.wav').exists()


def getCompletedPairs(output_dir):
    """Return set of pair_ids that already have degraded.wav."""
    pairs_dir = output_dir / 'pairs'
    if not pairs_dir.exists():
        return set()
    completed = set()
    for d in pairs_dir.iterdir():
        if d.is_dir() and (d / 'degraded.wav').exists():
            completed.add(int(d.name))
    return completed


# ══════════════════════════════════════════════════════════════════════════════
# Audio Utilities
# ══════════════════════════════════════════════════════════════════════════════

def crossfadeSegments(seg1, seg2, crossfade_samples):
    """Raised cosine crossfade between two adjacent segments."""
    if crossfade_samples <= 0:
        return np.concatenate([seg1, seg2])
    if len(seg1) < crossfade_samples or len(seg2) < crossfade_samples:
        return np.concatenate([seg1, seg2])

    t = np.linspace(0, np.pi, crossfade_samples, dtype=np.float32)
    fade_out = 0.5 * (1 + np.cos(t))
    fade_in = 0.5 * (1 - np.cos(t))

    crossfade = seg1[-crossfade_samples:] * fade_out + seg2[:crossfade_samples] * fade_in
    return np.concatenate([seg1[:-crossfade_samples], crossfade, seg2[crossfade_samples:]])


def segmentClip(audio, sr, clip_sec, crossfade_ms):
    """Segment long audio into fixed-length windows with crossfade."""
    clip_samples = int(sr * clip_sec)
    crossfade_samples = int(sr * crossfade_ms / 1000)
    total_samples = len(audio)

    if total_samples <= clip_samples:
        # Pad with zeros if too short
        padded = np.zeros(clip_samples, dtype=np.float32)
        padded[:total_samples] = audio[:total_samples]
        return [padded]

    clips = []
    step = clip_samples - crossfade_samples
    start = 0
    while start + clip_samples <= total_samples:
        clip = audio[start:start + clip_samples].copy()
        clips.append(clip)
        start += step

    # Handle last segment if it doesn't align perfectly
    if start < total_samples:
        remaining = total_samples - start
        if remaining >= clip_samples:
            clip = audio[start:start + clip_samples].copy()
        else:
            clip = np.zeros(clip_samples, dtype=np.float32)
            clip[:remaining] = audio[start:start + remaining]
        clips.append(clip)

    # Apply crossfade between adjacent clips
    if len(clips) > 1:
        result = [clips[0]]
        for i in range(1, len(clips)):
            result[-1] = crossfadeSegments(result[-1], clips[i], crossfade_samples)
        # The crossfadeSegments function combines segments, so we take the final result
        return [result[-1]] if len(result) == 1 else result

    return clips


def normalizeToTarget(audio, sr, target_lufs=-23.0):
    """Normalize audio to target LUFS with peak-clip safety."""
    # Simple RMS-based normalization as proxy for LUFS
    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
    if rms < 1e-10:
        return audio.astype(np.float32)

    # Convert target LUFS to approximate RMS
    # LUFS ≈ -0.691 + 10*log10(rms^2) for simple signals
    target_rms = 10 ** ((target_lufs + 0.691) / 20)
    gain = target_rms / rms

    normalized = audio.astype(np.float32) * gain

    # Peak-clip safety
    peak = np.max(np.abs(normalized))
    if peak > 0.99:
        normalized *= 0.99 / peak

    return normalized


def loadAllClips(pristine_paths, clusters):
    """Load, segment, and normalize all pristine audio for speakers in clusters."""
    all_clips = {}

    for speaker_id, info in clusters.items():
        clips = []
        # Determine source path based on speaker_id prefix
        if speaker_id.startswith('p225') or speaker_id.startswith('p226') or speaker_id.startswith('p227'):
            # VCTK speakers
            source_dir = pristine_paths['vctk']
        elif speaker_id.startswith('LJ'):
            # LJSpeech
            source_dir = pristine_paths['ljspeech']
        elif speaker_id.startswith('daps'):
            # DAPS
            source_dir = pristine_paths['daps']
        else:
            # Try VCTK as default
            source_dir = pristine_paths['vctk']

        # Glob audio files for this speaker
        if source_dir.exists():
            audio_files = sorted(source_dir.glob(f'**/{speaker_id}*.wav'))
            if not audio_files:
                # Try alternative patterns
                audio_files = sorted(source_dir.glob(f'**/*{speaker_id}*.wav'))

            for audio_path in audio_files:
                try:
                    audio, file_sr = sf.read(str(audio_path), dtype='float32')
                    # Ensure mono
                    if audio.ndim > 1:
                        audio = audio.mean(axis=1)
                    # Resample to target SR
                    if file_sr != SR:
                        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SR)
                    # Segment into clips
                    clip_list = segmentClip(audio, SR, CLIP_SEC, CROSSFADE_MS)
                    for clip in clip_list:
                        normalized = normalizeToTarget(clip, SR)
                        clips.append(normalized)
                except Exception as e:
                    print(f"  [WARN] Failed to load {audio_path}: {e}")
                    continue

        if clips:
            all_clips[speaker_id] = clips

    return all_clips


def saveAllClipsCache(all_clips, cache_path):
    """Save all_clips dict to npz cache."""
    save_dict = {}
    metadata = {}
    for speaker_id, clips in all_clips.items():
        for clip_idx, clip in enumerate(clips):
            key = f"{speaker_id}_{clip_idx}"
            save_dict[key] = clip
            metadata[key] = {'speaker_id': speaker_id, 'clip_idx': clip_idx}
    save_dict['_metadata'] = np.array(json.dumps(metadata))
    np.savez_compressed(str(cache_path), **save_dict)


def loadAllClipsCache(cache_path, clusters):
    """Load all_clips dict from npz cache."""
    data = np.load(str(cache_path), allow_pickle=True)
    metadata = json.loads(str(data['_metadata']))
    all_clips = defaultdict(list)
    for key, info in metadata.items():
        if key == '_metadata':
            continue
        speaker_id = info['speaker_id']
        if speaker_id in clusters:
            all_clips[speaker_id].append(data[key])
    return dict(all_clips)


# ══════════════════════════════════════════════════════════════════════════════
# Random Parameter Generators
# ══════════════════════════════════════════════════════════════════════════════

def randomEqBands():
    """Generate 1-6 random EQ bands with clustered frequency rejection."""
    n_bands = random.randint(1, 6)
    bands = []
    freqs_used = []

    for _ in range(n_bands):
        retries = 0
        while retries < 20:
            freq = np.exp(random.uniform(np.log(20), np.log(20000)))
            # Check for clustering
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
            'filter_type': random.choice(['peak', 'low_shelf', 'high_shelf',
                                          'highpass', 'lowpass', 'bandpass', 'notch']),
            'stereo_skew_db': round(random.uniform(-6, 6), 2),
            'dynamic_depth': round(random.uniform(0, 1), 2),
        }
        bands.append(band)
    return bands


def randomCompressorParams():
    """Generate random compressor parameters."""
    return {
        'threshold_db': round(random.uniform(-60, 0), 2),
        'ratio': round(random.uniform(1, 20), 2),
        'attack_ms': round(random.uniform(0.1, 100), 2),
        'release_ms': round(random.uniform(10, 1000), 2),
        'knee_db': round(random.uniform(0, 12), 2),
        'lookahead_ms': round(random.uniform(0, 10), 2),
        'hold_ms': round(random.uniform(0, 200), 2),
        'wet_dry_mix': round(random.uniform(0, 1), 2),
        'stereo_link': round(random.uniform(0, 1), 2),
        'sidechain_hp_hz': round(random.uniform(20, 500), 1),
        'sidechain_lp_hz': round(random.uniform(500, 20000), 1),
        'saturate_drive_db': round(random.uniform(0, 12), 2),
        'output_trim_db': round(random.uniform(-12, 12), 2),
        'detector_type': random.choice([0, 1, 2, 3]),
    }


def randomEsserParams():
    """Generate random esser parameters."""
    return {
        'center_freq_hz': round(random.uniform(4000, 10000), 1),
        'threshold_db': round(random.uniform(-60, 0), 2),
        'ratio': round(random.uniform(0.25, 20), 2),
        'bandwidth_hz': round(random.uniform(500, 4000), 1),
        'attack_ms': round(random.uniform(0.1, 50), 2),
        'release_ms': round(random.uniform(10, 500), 2),
    }


def randomSaturatorParams():
    """Generate random saturator parameters."""
    return {
        'drive_db': round(random.uniform(0, 24), 2),
        'mix': round(random.uniform(0, 1), 2),
        'sat_type': random.choice(['tube', 'tape', 'diode', 'asymmetric']),
        'hpf_hz': round(random.uniform(20, 500), 1),
        'lpf_hz': round(random.uniform(2000, 20000), 1),
        'oversampling': random.choice([1, 2, 4]),
        'output_trim_db': round(random.uniform(-12, 12), 2),
    }


def randomLimiterParams():
    """Generate random limiter parameters."""
    return {
        'ceiling_db': round(random.uniform(-12, 0), 2),
        'release_ms': round(random.uniform(1, 500), 2),
        'lookahead_ms': round(random.uniform(0, 10), 2),
        'clip_mode': random.choice(['soft', 'hard']),
        'stereo_link': round(random.uniform(0, 1), 2),
        'oversampling': random.choice([1, 2, 4]),
    }


def randomTransientParams():
    """Generate random transient shaper parameters."""
    return {
        'attack_gain_db': round(random.uniform(-24, 24), 2),
        'sustain_gain_db': round(random.uniform(-24, 24), 2),
        'attack_time_ms': round(random.uniform(0.1, 50), 2),
        'release_time_ms': round(random.uniform(10, 500), 2),
        'sensitivity_db': round(random.uniform(-30, 0), 2),
        'mix': round(random.uniform(0, 1), 2),
    }


def randomGainParams():
    """Generate random gain parameters."""
    return {
        'gain_db': round(random.uniform(-12, 12), 2),
        'stereo_balance': round(random.uniform(-1, 1), 2),
    }


def generateDegradationParams():
    """Generate random parameters for all 7 plugins."""
    return {
        'eq_bands': randomEqBands(),
        'compressor': randomCompressorParams(),
        'esser': randomEsserParams(),
        'saturator': randomSaturatorParams(),
        'limiter': randomLimiterParams(),
        'transient': randomTransientParams(),
        'gain': randomGainParams(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Degradation Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def applyDegradation(audio, sr, params):
    """Apply sequential 7-plugin degradation pipeline to audio."""
    audio = audio.astype(np.float32)

    # 1. EQ
    audio, _ = equalizer.process(audio, sr, bands=params['eq_bands'])

    # 2. Compressor
    audio, _ = compressor.process(audio, sr, **params['compressor'])

    # 3. Esser
    audio, _ = esser.process(audio, sr, **params['esser'])

    # 4. Saturator
    audio, _ = saturator.process(audio, sr, **params['saturator'])

    # 5. Limiter
    audio, _ = limiter.process(audio, sr, **params['limiter'])

    # 6. Transient Shaper
    audio, _ = transient.process(audio, sr, **params['transient'])

    # 7. Gain
    audio, _ = gain1.process(audio, sr, **params['gain'])

    return audio


# ══════════════════════════════════════════════════════════════════════════════
# Pair Generation
# ══════════════════════════════════════════════════════════════════════════════

def generatePairs(all_clips, clusters, n_pairs):
    """Generate speaker pairs within clusters for degraded/reference pairing."""
    # Group speakers by cluster
    cluster_speakers = defaultdict(list)
    for sid, info in clusters.items():
        if sid in all_clips and len(all_clips[sid]) > 0:
            cluster_speakers[info['cluster']].append(sid)

    pairs = []
    source_counts = defaultdict(int)
    ref_counts = defaultdict(int)

    # Round-robin through cluster speaker pairs
    for cluster_id, speakers in cluster_speakers.items():
        if len(speakers) < 2:
            continue  # Need at least 2 speakers for pairing

        # Create all unordered speaker pairs within this cluster
        speaker_pairs = []
        for i in range(len(speakers)):
            for j in range(i + 1, len(speakers)):
                speaker_pairs.append((speakers[i], speakers[j]))

        # Distribute pairs across clusters proportionally
        cluster_n = max(1, int(n_pairs * len(speakers) / 130))

        for _ in range(cluster_n):
            src_spk, ref_spk = random.choice(speaker_pairs)
            # Pick random clips
            src_idx = random.randint(0, len(all_clips[src_spk]) - 1)
            ref_idx = random.randint(0, len(all_clips[ref_spk]) - 1)

            pairs.append({
                'pair_id': len(pairs) + 1,
                'source_speaker': src_spk,
                'ref_speaker': ref_spk,
                'cluster_id': cluster_id,
                'src_clip_idx': src_idx,
                'ref_clip_idx': ref_idx,
            })

            source_counts[src_spk] += 1
            ref_counts[ref_spk] += 1

    # Trim or pad to exact n_pairs
    random.shuffle(pairs)
    pairs = pairs[:n_pairs]

    return pairs, source_counts, ref_counts


# ══════════════════════════════════════════════════════════════════════════════
# Save Functions
# ══════════════════════════════════════════════════════════════════════════════

def makeJsonSafe(obj):
    """Recursively convert numpy types to Python native types for JSON."""
    if isinstance(obj, dict):
        return {k: makeJsonSafe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [makeJsonSafe(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def savePair(pair_id, degraded, reference, params, output_dir):
    """Save a single degraded/reference pair to disk."""
    pair_dir = output_dir / 'pairs' / f'{pair_id:08d}'
    pair_dir.mkdir(parents=True, exist_ok=True)

    # Write audio
    degraded_clip = np.clip(degraded, -1.0, 1.0)
    reference_clip = np.clip(reference, -1.0, 1.0)

    sf.write(str(pair_dir / 'degraded.wav'),
             (degraded_clip * 32767).astype(np.int16), SR)
    sf.write(str(pair_dir / 'reference.wav'),
             (reference_clip * 32767).astype(np.int16), SR)

    # Write params (JSON-serializable)
    params_clean = makeJsonSafe(params)
    with open(pair_dir / 'params.json', 'w') as f:
        json.dump(params_clean, f, indent=2)


def saveMetadata(clusters, identity_floors, pairs, source_counts, ref_counts, output_dir):
    """Save global metadata and degradation params."""
    metadata = {
        'version': 1,
        'n_pairs': len(pairs),
        'sample_rate': SR,
        'clip_seconds': CLIP_SEC,
        'cluster_floors': identity_floors,
        'cluster_assignments': {sid: info['cluster'] for sid, info in clusters.items()},
        'n_speakers': len(clusters),
        'source_balance': dict(source_counts),
        'ref_balance': dict(ref_counts),
    }
    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    # Also save full degradation params for debugging
    all_params = {}
    for pair in pairs:
        all_params[str(pair['pair_id'])] = pair
    with open(output_dir / 'degradation_params.json', 'w') as f:
        json.dump(all_params, f, indent=2)


def getOutputSizeGB():
    """Calculate total output size in GB."""
    total = sum(f.stat().st_size for f in OUTPUT.rglob('*') if f.is_file())
    return total / (1024 ** 3)