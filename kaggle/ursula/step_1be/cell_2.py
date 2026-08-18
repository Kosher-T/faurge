# ══════════════════════════════════════════════════════════════════════════════
# Step 2 — Load Audio (DAPS + VCTK, Dynamic Speaker Selection)
# ══════════════════════════════════════════════════════════════════════════════
# Scan both DAPS and VCTK datasets. Pick speakers at random.
# VCTK has no gender tags — just speaker directories like p225, p226.
# Extract middle 5s to avoid clip beginning silence.

import soundfile as sf
import librosa

def load_audio_middle(path, target_samples=CLIP_SAMPLES):
    """Load audio, extract middle `target_samples` samples."""
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

def select_speakers(daps_dir, vctk_dir, n_train=N_TRAIN_CLIPS, n_test=N_TEST_CLIPS):
    """Scan DAPS + VCTK, pick n_train + n_test speakers at random.
    Returns (train_paths, test_paths, train_speakers, test_speakers)."""
    speakers = {}  # {speaker_name: [path, ...]}

    # DAPS: speakers start with m/f (e.g. m001_script1_norm.wav)
    if daps_dir.exists():
        for wav in daps_dir.rglob('*.wav'):
            speaker = wav.stem.split('_script')[0]
            speakers.setdefault(speaker, []).append(wav)
        print(f"DAPS speakers found: {len([s for s in speakers if s.startswith(('m', 'f'))])}")

    # VCTK: directories like p225, p226, etc.
    if vctk_dir.exists():
        vctk_count = 0
        for speaker_dir in sorted(vctk_dir.iterdir()):
            if speaker_dir.is_dir():
                wav_files = list(speaker_dir.glob('*.wav'))
                if wav_files:
                    speakers[speaker_dir.name] = wav_files
                    vctk_count += 1
        print(f"VCTK speakers found: {vctk_count}")

    all_speakers = list(speakers.keys())
    print(f"Total speakers: {len(all_speakers)}")

    np.random.seed(42)
    np.random.shuffle(all_speakers)

    selected = all_speakers[:n_train + n_test]
    train_speakers = selected[:n_train]
    test_speakers = selected[n_train:]

    # Pick 1 random clip per speaker
    train_paths = [np.random.choice(speakers[s]) for s in train_speakers]
    test_paths = [np.random.choice(speakers[s]) for s in test_speakers]

    return train_paths, test_paths, train_speakers, test_speakers

train_paths, test_paths, train_speakers, test_speakers = select_speakers(DAPS_DIR, VCTK_DIR)

print(f"\nTrain speakers ({len(train_speakers)}): {train_speakers}")
print(f"Test speakers ({len(test_speakers)}): {test_speakers}")

# ── Load clips ───────────────────────────────────────────────────────────────
train_clips = []
train_clip_names = []
for path, speaker in zip(train_paths, train_speakers):
    audio = load_audio_middle(path)
    if audio is not None:
        train_clips.append(audio)
        train_clip_names.append(f"{speaker}_{path.stem}")
        print(f"  Train: {speaker} — {path.name} — {len(audio)/SR:.1f}s")

test_clips = []
test_clip_names = []
for path, speaker in zip(test_paths, test_speakers):
    audio = load_audio_middle(path)
    if audio is not None:
        test_clips.append(audio)
        test_clip_names.append(f"{speaker}_{path.stem}")
        print(f"  Test:  {speaker} — {path.name} — {len(audio)/SR:.1f}s")

all_clips = train_clips + test_clips
all_clip_names = train_clip_names + test_clip_names
print(f"\nLoaded {len(train_clips)} train + {len(test_clips)} test = {len(all_clips)} clips")
