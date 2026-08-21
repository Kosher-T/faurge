# ══════════════════════════════════════════════════════════════════════════════
# Step 2a — Load Audio (5 male, 5 female)
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

def find_clips_balanced(pristine_dir, male_count=5, female_count=5):
    wav_files = sorted(list(pristine_dir.rglob('*.wav')))
    if not wav_files:
        raise FileNotFoundError(f"No .wav files found in {pristine_dir}")

    speakers = {}
    for path in wav_files:
        stem = path.stem
        speaker = stem.split('_script')[0]
        if speaker not in speakers:
            speakers[speaker] = []
        speakers[speaker].append(path)

    male_speakers = sorted([s for s in speakers if s.startswith('m')])
    female_speakers = sorted([s for s in speakers if s.startswith('f')])

    print(f"Male speakers:   {len(male_speakers)}")
    print(f"Female speakers: {len(female_speakers)}")

    selected = []
    np.random.seed(42)

    male_pick = np.random.choice(male_speakers, size=min(male_count, len(male_speakers)), replace=False)
    for speaker in male_pick:
        selected.append(speakers[speaker][0])

    female_pick = np.random.choice(female_speakers, size=min(female_count, len(female_speakers)), replace=False)
    for speaker in female_pick:
        selected.append(speakers[speaker][0])

    np.random.shuffle(selected)
    return selected

clip_paths = find_clips_balanced(PRISTINE, male_count=CLIPS_PER_GENDER, female_count=CLIPS_PER_GENDER)
clean_clips = []
clip_names = []

for path in clip_paths:
    audio = load_audio_middle(path)
    if audio is not None:
        clean_clips.append(audio)
        clip_names.append(path.stem)
        print(f"Loaded: {path.name} — {len(audio)/SR:.1f}s")

print(f"\nLoaded {len(clean_clips)} clips")
