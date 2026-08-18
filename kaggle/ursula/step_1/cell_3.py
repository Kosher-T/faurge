# ══════════════════════════════════════════════════════════════════════════════
# Step 1 — Causality: Load Audio
# ══════════════════════════════════════════════════════════════════════════════
# Load N_CLIPS voice clips from daps-pristine.
# Filters by CLIP_PREFIX (e.g., 'm' for male clips).
# Takes 5s from the middle to avoid silence at start.

import soundfile as sf
import librosa

# ── Audio Loading ─────────────────────────────────────────────────────────────

def load_audio_middle(path, target_samples=CLIP_SAMPLES):
    """Load audio and take target_samples from the middle (avoid silence at start)."""
    audio, file_sr = sf.read(str(path), dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != SR:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SR)
    if len(audio) < SR * 1.0:  # Need at least 1 second
        return None
    # Take from the middle
    start = (len(audio) - target_samples) // 2
    if start < 0:
        # Clip is shorter than target — pad with zeros
        padded = np.zeros(target_samples, dtype=np.float32)
        padded[:len(audio)] = audio
        return padded
    return audio[start:start + target_samples]

# ── Find clips with prefix filter ─────────────────────────────────────────────

def find_clips(pristine_dir, prefixes=CLIP_PREFIXES, max_clips=N_CLIPS):
    """Find clips starting with any of the prefixes (e.g., ['m', 'f'] for mixed)."""
    wav_files = sorted(list(pristine_dir.rglob('*.wav')))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {pristine_dir}")

    # Filter by prefixes
    filtered = [p for p in wav_files if any(p.stem.startswith(px) for px in prefixes)]
    print(f"Found {len(filtered)} clips with prefixes {prefixes} (out of {len(wav_files)} total)")

    # Group by speaker
    speakers = {}
    for path in filtered:
        stem = path.stem
        speaker = stem.split('_script')[0]
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append(path)

    print(f"Speakers available: {list(speakers.keys())}")

    # Pick one clip from RANDOM speakers
    selected = []
    speaker_list = list(speakers.keys())
    np.random.shuffle(speaker_list)
    for speaker in speaker_list:
        if len(selected) >= max_clips:
            break
        selected.append(speakers[speaker][0])

    return selected

# ── Load clips ────────────────────────────────────────────────────────────────

clip_paths = find_clips(PRISTINE, prefixes=CLIP_PREFIXES, max_clips=N_CLIPS)
clean_clips = []
clip_names = []

for path in clip_paths:
    audio = load_audio_middle(path)
    if audio is not None:
        clean_clips.append(audio)
        clip_names.append(path.stem)
        print(f"Loaded: {path.name} — {len(audio)/SR:.1f}s, {len(audio)} samples")

print(f"\nLoaded {len(clean_clips)} clips from {len(set(clip_names))} speakers")
