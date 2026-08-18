# ══════════════════════════════════════════════════════════════════════════════
# Step 1bb — Load Audio (1 male, 1 female)
# ══════════════════════════════════════════════════════════════════════════════

import soundfile as sf
import librosa

def load_audio_middle(path, target_samples=CLIP_SAMPLES):
    audio, file_sr = sf.read(str(path), dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if file_sr != SR:
        audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SR)
    if len(audio) < SR * 1.0:
        return None
    start = (len(audio) - target_samples) // 2
    if start < 0:
        padded = np.zeros(target_samples, dtype=np.float32)
        padded[:len(audio)] = audio
        return padded
    return audio[start:start + target_samples]

def find_clips(pristine_dir, prefixes=CLIP_PREFIXES, max_clips=N_CLIPS):
    wav_files = sorted(list(pristine_dir.rglob('*.wav')))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {pristine_dir}")

    filtered = [p for p in wav_files if any(p.stem.startswith(px) for px in prefixes)]

    speakers = {}
    for path in filtered:
        stem = path.stem
        speaker = stem.split('_script')[0]
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append(path)

    # Pick exactly 1 per prefix
    selected = []
    for px in prefixes:
        matching = [s for s in speakers if s.startswith(px)]
        if matching:
            speaker = np.random.choice(matching)
            selected.append(speakers[speaker][0])

    return selected

clip_paths = find_clips(PRISTINE, prefixes=CLIP_PREFIXES, max_clips=N_CLIPS)
clean_clips = []
clip_names = []

for path in clip_paths:
    audio = load_audio_middle(path)
    if audio is not None:
        clean_clips.append(audio)
        clip_names.append(path.stem)
        print(f"Loaded: {path.name} — {len(audio)/SR:.1f}s")

print(f"\nLoaded {len(clean_clips)} clips: {clip_names}")
