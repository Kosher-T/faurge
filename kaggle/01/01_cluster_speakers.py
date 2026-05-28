# %% [markdown]
# # Phase 1: Cluster Analysis & Identity Floor
#
# Clusters speakers by voice similarity using 67D physical metrics (LTAS + LUFS + Dynamic Range).
# Outputs cluster assignments, centroids with thresholds, and identity floors for Ursula's RL training.
#
# **Inputs:** Pristine audio from VCTK, LJSpeech, DAPS
# **Outputs:** `clusters.json`, `cluster_centroids.json`, `identity_floors.json`

# %%
# ═══════════════════════════════════════════════════════════════════════
# CELL 1 — Setup
# ═══════════════════════════════════════════════════════════════════════

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

print(f"SR={SR}, CLIP_SEC={CLIP_SEC}, CLIP_SAMPLES={CLIP_SAMPLES}")
print(f"Samples per speaker={SAMPLES_PER_SPEAKER}, Clusters={N_CLUSTERS}")
print(f"Output: {CLUSTER_DATA_DIR}")

# %% [markdown]
# # All Functions
#
# 67D metrics extractor, speaker discovery, clustering, identity floors, validation, and output.

# %%
# ═══════════════════════════════════════════════════════════════════════
# CELL 2 — All Functions
# ═══════════════════════════════════════════════════════════════════════

# ─── Bark Scale Utilities ────────────────────────────────────────────

def hz_to_bark(hz: float) -> float:
    """Convert Hz to Bark scale (Zwicker / Trapezoidal)."""
    return 13.0 * np.arctan(0.00076 * hz) + 3.5 * np.arctan((hz / 7500.0) ** 2)


def bark_to_hz(bark: float) -> float:
    """Convert Bark scale back to Hz."""
    return 600.0 * np.sinh(bark / 7.0)


def create_bark_filterbank(
    n_bands: int = BARK_N_BANDS,
    low_hz: float = BARK_LOW_HZ,
    high_hz: float = BARK_HIGH_HZ,
    n_fft: int = FFT_SIZE,
    sr: int = SR,
) -> np.ndarray:
    """
    Create triangular Bark-scale filterbank.
    Returns (n_bands, n_fft//2+1) weight matrix.
    """
    bark_low = hz_to_bark(low_hz)
    bark_high = hz_to_bark(high_hz)
    bark_centers = np.linspace(bark_low, bark_high, n_bands)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)  # (n_fft//2+1,)

    filterbank = np.zeros((n_bands, len(freqs)), dtype=np.float64)
    for i in range(n_bands):
        center = bark_centers[i]
        if i == 0:
            left_edge = bark_low
        else:
            left_edge = (bark_centers[i - 1] + center) / 2.0
        if i == n_bands - 1:
            right_edge = bark_high
        else:
            right_edge = (center + bark_centers[i + 1]) / 2.0

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


# Pre-compute filterbank once
_BARK_FILTERBANK = create_bark_filterbank()

# ─── LUFS (ITU-R BS.1770-4) ─────────────────────────────────────────

def _design_k_weighting_filters(sr: int = SR):
    """
    Design the two stages of K-weighting filters per ITU-R BS.1770-4.
    Returns (sos_pre, sos_rlb) as second-order-sections.
    """
    from scipy.signal import butter

    # Stage 1: Pre-filter (compensates for acoustic head effect)
    # High-shelf at ~1500 Hz with +4 dB gain
    # Simplified: 1st-order HPF at 1500 Hz
    sos_pre = butter(1, 1500.0, btype="high", fs=sr, output="sos")

    # Stage 2: RLB weighting (Revised Low-frequency B-curve)
    # High-shelf boost starting around 4000 Hz
    # Implemented as a 2nd-order high-shelf
    f0 = 4000.0
    gain_db = 3.0
    gain_lin = 10.0 ** (gain_db / 20.0)
    # Simple approach: use a high-pass filter to approximate the shelf
    sos_rlb = butter(2, f0, btype="high", fs=sr, output="sos")

    return sos_pre, sos_rlb


def compute_lufs_1d(audio: np.ndarray, sr: int = SR) -> float:
    """
    Compute integrated LUFS per ITU-R BS.1770-4.
    Two-pass gating algorithm: absolute gate at -70 LUFS, relative gate at -10 dB.
    """
    from scipy.signal import sosfilt

    if len(audio) < sr * 0.1:
        # Too short for meaningful gating — compute ungated
        filtered = sosfilt(_K_WEIGHTING_SOS, audio)
        mean_sq = np.mean(filtered ** 2)
        lufs = -0.691 + 10.0 * np.log10(mean_sq + 1e-20)
        return float(lufs)

    # Apply K-weighting
    filtered = audio.copy()
    for sos in _K_WEIGHTING_SOS:
        filtered = sosfilt(sos, filtered)

    # Block integration: 400ms blocks, 75% overlap (100ms hop)
    block_size = int(0.4 * sr)  # 19200 samples
    hop_size = int(0.1 * sr)    # 4800 samples
    n_blocks = max(1, (len(filtered) - block_size) // hop_size + 1)

    block_powers = np.zeros(n_blocks)
    for i in range(n_blocks):
        start = i * hop_size
        end = start + block_size
        if end > len(filtered):
            break
        block = filtered[start:end]
        block_powers[i] = np.mean(block ** 2)

    # Absolute gate at -70 LUFS
    abs_gate = 10.0 ** ((-70.0 + 0.691) / 10.0)
    gated_mask = block_powers >= abs_gate

    if not np.any(gated_mask):
        return -70.0

    # Relative gate at -10 dB below absolute-gated mean
    abs_gated_mean = np.mean(block_powers[gated_mask])
    rel_threshold = abs_gated_mean * (10.0 ** (-10.0 / 10.0))
    rel_gated_mask = block_powers >= rel_threshold

    if not np.any(rel_gated_mask):
        return -70.0

    # Final integrated loudness
    final_mean = np.mean(block_powers[rel_gated_mask])
    lufs = -0.691 + 10.0 * np.log10(final_mean + 1e-20)
    return float(lufs)


# Design K-weighting filters once
_K_WEIGHTING_SOS = _design_k_weighting_filters(SR)

# ─── Crest Factor & ZCR ──────────────────────────────────────────────

def compute_crest_factor(audio: np.ndarray) -> float:
    """Crest Factor in dB: 20*log10(peak/rms)."""
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2))
    return 20.0 * np.log10(peak / (rms + 1e-10))


def compute_zcr(audio: np.ndarray) -> float:
    """Zero-Crossing Rate."""
    signs = np.sign(audio)
    signs[signs == 0] = 1
    crossings = np.sum(np.abs(np.diff(signs)) > 0)
    return float(crossings / (2.0 * (len(audio) - 1)))

# ─── 67D Metrics Extractor ───────────────────────────────────────────

def compute_ltas_64d(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Compute LTAS: 64D Bark-scale spectral envelope.
    STFT → power spectrum → Bark filterbank → dB → time average.
    """
    from scipy.signal import stft

    # STFT
    f, t, Zxx = stft(audio, fs=sr, nperseg=FFT_SIZE, noverlap=FFT_SIZE - HOP_SIZE,
                     window="hann", return_onesided=True)
    power = np.abs(Zxx) ** 2  # (n_freqs, n_frames)

    # Apply Bark filterbank → (64, n_frames)
    band_energy = _BARK_FILTERBANK @ power  # (64, n_frames)

    # Time-average
    mean_energy = np.mean(band_energy, axis=1)  # (64,)

    # Convert to dB
    ltas = 10.0 * np.log10(mean_energy + 1e-10)

    return ltas.astype(np.float64)


def extract_metrics_67d(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """
    Extract full 67D metric vector: LTAS(64) + LUFS(1) + Crest(1) + ZCR(1).
    """
    ltas = compute_ltas_64d(audio, sr)      # (64,)
    lufs = np.array([compute_lufs_1d(audio, sr)])  # (1,)
    crest = np.array([compute_crest_factor(audio)])  # (1,)
    zcr = np.array([compute_zcr(audio)])            # (1,)
    return np.concatenate([ltas, lufs, crest, zcr])  # (67,)

# ─── Speaker Discovery ───────────────────────────────────────────────

def discover_speakers(paths: dict) -> dict[str, list[Path]]:
    """
    Discover all speakers across VCTK, LJSpeech, DAPS.
    Returns {speaker_id: [Path, ...]}.
    """
    speakers = defaultdict(list)

    # VCTK: subfolders per speaker
    vctk_root = paths["vctk"]
    if vctk_root.exists():
        for speaker_dir in sorted(vctk_root.iterdir()):
            if speaker_dir.is_dir():
                wav_files = sorted(speaker_dir.glob("*.wav"))
                if wav_files:
                    speakers[speaker_dir.name].extend(wav_files)

    # LJSpeech: single speaker
    ljspeech_root = paths["ljspeech"]
    if ljspeech_root.exists():
        wav_files = sorted(ljspeech_root.glob("*.wav"))
        if wav_files:
            speakers["LJ001"].extend(wav_files)

    # DAPS: filename pattern [speaker]_[script]_[device]_[room].wav
    daps_root = paths["daps"]
    if daps_root.exists():
        for wav_file in sorted(daps_root.rglob("*.wav")):
            speaker_id = wav_file.stem.split("_")[0]
            speakers[speaker_id].append(wav_file)

    return dict(speakers)

# ─── Speaker Profile Computation ─────────────────────────────────────

def compute_speaker_profiles(
    speakers: dict[str, list[Path]],
    samples_per_speaker: int = SAMPLES_PER_SPEAKER,
) -> dict[str, np.ndarray]:
    """
    For each speaker, sample clips, extract 67D per clip, average → speaker profile.
    For DAPS, segment long clips into 5s windows first.
    """
    profiles = {}
    total_speakers = len(speakers)

    for idx, (speaker_id, clip_paths) in enumerate(speakers.items(), 1):
        t0 = time.time()

        # For DAPS, segment long clips into 5s windows
        all_segments = []
        for p in clip_paths:
            try:
                audio, file_sr = sf.read(p, dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                if file_sr != SR:
                    audio = librosa.resample(audio, orig_sr=file_sr, target_sr=SR)
            except Exception as e:
                print(f"  [WARN] Failed to load {p.name}: {e}")
                continue

            if len(audio) < SR * 0.5:
                continue

            # Segment into 5s windows
            for start in range(0, len(audio) - SR, CLIP_SAMPLES):
                chunk = audio[start : start + CLIP_SAMPLES]
                if len(chunk) < CLIP_SAMPLES:
                    chunk = np.pad(chunk, (0, CLIP_SAMPLES - len(chunk)))
                rms = np.sqrt(np.mean(chunk ** 2))
                if rms > 1e-4:
                    all_segments.append(chunk)

        if not all_segments:
            print(f"  [WARN] No valid segments for {speaker_id}")
            continue

        # Sample clips
        n_sample = min(samples_per_speaker, len(all_segments))
        indices = sorted(random.sample(range(len(all_segments)), n_sample))
        sampled = [all_segments[i] for i in indices]

        # Extract 67D per clip
        metrics_list = []
        for clip in sampled:
            m = extract_metrics_67d(clip, SR)
            metrics_list.append(m)

        # Average → speaker profile
        profile = np.mean(metrics_list, axis=0)
        profiles[speaker_id] = profile

        elapsed = time.time() - t0
        print(f"  [{idx}/{total_speakers}] {speaker_id}: {n_sample} clips → "
              f"67D profile ({elapsed:.1f}s)")

    return profiles

# ─── Clustering ───────────────────────────────────────────────────────

def run_clustering(
    speaker_profiles: dict[str, np.ndarray],
    n_clusters: int = N_CLUSTERS,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    K-means clustering on speaker profiles.
    Returns (labels, centroids_67d_dict, thresholds_dict, speaker_ids, X, X_scaled).
    """
    speaker_ids = sorted(speaker_profiles.keys())
    X = np.array([speaker_profiles[sid] for sid in speaker_ids])  # (n_speakers, 67)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Try k=4,6,8 for silhouette evaluation
    print("\n── Silhouette Score Comparison ──")
    for k in [4, 6, 8]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_k = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels_k)
        print(f"  k={k}: silhouette = {score:.4f}")

    # Final clustering with target k
    km_final = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km_final.fit_predict(X_scaled)

    # Compute centroids (in original 67D space for interpretability)
    centroids_67d_dict = {}
    thresholds_dict = {}

    for c in range(n_clusters):
        mask = labels == c
        centroid_67d = np.mean(X[mask], axis=0)  # mean in original space

        # 95th percentile of within-cluster pairwise MSE (original space)
        cluster_profiles = X[mask]
        if len(cluster_profiles) > 1:
            pairwise_mse = []
            for i in range(len(cluster_profiles)):
                for j in range(i + 1, len(cluster_profiles)):
                    mse = np.mean((cluster_profiles[i] - cluster_profiles[j]) ** 2)
                    pairwise_mse.append(mse)
            threshold = float(np.percentile(pairwise_mse, 95))
        else:
            threshold = 0.0

        centroids_67d_dict[f"cluster_{c}"] = centroid_67d
        thresholds_dict[f"cluster_{c}"] = threshold

    # Print cluster distribution
    print(f"\n── Cluster Assignments ──")
    for c in range(n_clusters):
        members = [speaker_ids[i] for i in range(len(labels)) if labels[i] == c]
        print(f"  Cluster {c} ({len(members)} speakers): "
              f"{', '.join(members[:10])}{'...' if len(members) > 10 else ''}")

    return labels, centroids_67d_dict, thresholds_dict, speaker_ids, X, X_scaled

# ─── Identity Floors ─────────────────────────────────────────────────

def compute_identity_floors(
    speaker_profiles: dict[str, np.ndarray],
    cluster_labels: np.ndarray,
    speaker_ids: list[str],
) -> dict:
    """
    Compute within-cluster and cross-cluster pairwise MSE identity floors.
    """
    X = np.array([speaker_profiles[sid] for sid in speaker_ids])
    n_clusters = len(set(cluster_labels))
    floors = {}

    # Within-cluster floors
    print("\n── Within-Cluster Identity Floors ──")
    for c in range(n_clusters):
        mask = np.where(cluster_labels == c)[0]
        if len(mask) < 2:
            floors[f"cluster_{c}"] = 0.0
            continue

        cluster_profiles = X[mask]
        pairwise_mse = []
        for i in range(len(cluster_profiles)):
            for j in range(i + 1, len(cluster_profiles)):
                mse = np.mean((cluster_profiles[i] - cluster_profiles[j]) ** 2)
                pairwise_mse.append(mse)
        avg_mse = float(np.mean(pairwise_mse))
        floors[f"cluster_{c}"] = avg_mse
        print(f"  Cluster {c}: {avg_mse:.6f} (n={len(mask)}, pairs={len(pairwise_mse)})")

    # Cross-cluster floors
    floors["cross"] = {}
    print("\n── Cross-Cluster Identity Floors ──")
    for c1 in range(n_clusters):
        for c2 in range(c1 + 1, n_clusters):
            mask1 = np.where(cluster_labels == c1)[0]
            mask2 = np.where(cluster_labels == c2)[0]
            if len(mask1) == 0 or len(mask2) == 0:
                continue

            profiles1 = X[mask1]
            profiles2 = X[mask2]
            cross_mse = []
            for p1 in profiles1:
                for p2 in profiles2:
                    mse = float(np.mean((p1 - p2) ** 2))
                    cross_mse.append(mse)
            avg_cross = float(np.mean(cross_mse))
            key = f"cluster_{c1}_cluster_{c2}"
            floors["cross"][key] = avg_cross
            print(f"  {key}: {avg_cross:.6f}")

    return floors

# ─── Validation ───────────────────────────────────────────────────────

def validate_results(
    cluster_labels: np.ndarray,
    identity_floors: dict,
    centroids_67d: dict,
    thresholds: dict,
):
    """Run sanity checks on clustering results."""
    print("\n═══════════════════════════════════════════")
    print("  VALIDATION")
    print("═══════════════════════════════════════════")
    n_clusters = len(set(cluster_labels))
    all_pass = True

    # 1. Within-cluster floors < cross-cluster floors
    within_vals = [identity_floors[f"cluster_{c}"] for c in range(n_clusters)
                   if f"cluster_{c}" in identity_floors]
    cross_vals = list(identity_floors.get("cross", {}).values())
    if within_vals and cross_vals:
        avg_within = np.mean(within_vals)
        avg_cross = np.mean(cross_vals)
        check = avg_within < avg_cross
        status = "PASS" if check else "FAIL"
        if not check:
            all_pass = False
        print(f"  [{status}] Within-cluster avg ({avg_within:.6f}) < "
              f"cross-cluster avg ({avg_cross:.6f})")
    else:
        print("  [SKIP] Not enough data for floor comparison")

    # 2. All floors positive
    all_positive = all(v > 0 for v in within_vals) and all(v > 0 for v in cross_vals)
    status = "PASS" if all_positive else "FAIL"
    if not all_positive:
        all_pass = False
    print(f"  [{status}] All identity floors positive")

    # 3. Centroids well-separated (centroid-to-centroid MSE > within-cluster MSE)
    centroid_vecs = np.array([centroids_67d[f"cluster_{c}"] for c in range(n_clusters)])
    centroid_mses = []
    for i in range(n_clusters):
        for j in range(i + 1, n_clusters):
            mse = np.mean((centroid_vecs[i] - centroid_vecs[j]) ** 2)
            centroid_mses.append(mse)
    if centroid_mses and within_vals:
        min_centroid_mse = min(centroid_mses)
        max_within = max(within_vals)
        check = min_centroid_mse > max_within
        status = "PASS" if check else "FAIL"
        if not check:
            all_pass = False
        print(f"  [{status}] Min centroid separation ({min_centroid_mse:.6f}) > "
              f"max within-cluster floor ({max_within:.6f})")
    else:
        print("  [SKIP] Not enough data for centroid separation check")

    # 4. Thresholds positive and less than inter-centroid distances
    if centroid_mses:
        min_centroid_dist = min(centroid_mse for centroid_mse in centroid_mses)
        threshold_vals = list(thresholds.values())
        thresholds_valid = all(t > 0 and t < min_centroid_dist for t in threshold_vals)
        status = "PASS" if thresholds_valid else "FAIL"
        if not thresholds_valid:
            all_pass = False
        print(f"  [{status}] All thresholds (0, {min_centroid_dist:.6f})")
    else:
        print("  [SKIP] Not enough data for threshold check")

    # 5. Cluster assignments intuitive (manual check — just report)
    print(f"\n  Assignment summary: {n_clusters} clusters, "
          f"{len(cluster_labels)} speakers assigned")
    print(f"  Cluster sizes: {[int(np.sum(cluster_labels == c)) for c in range(n_clusters)]}")

    print(f"\n  Overall: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")

# ─── Save Outputs ─────────────────────────────────────────────────────

def save_outputs(
    speaker_profiles: dict[str, np.ndarray],
    cluster_labels: np.ndarray,
    centroids_67d: dict,
    thresholds: dict,
    identity_floors: dict,
    speaker_ids: list[str],
    output_dir: Path,
):
    """Save all artifacts to JSON files."""
    # clusters.json
    clusters = {}
    for idx, sid in enumerate(speaker_ids):
        clusters[sid] = {
            "cluster": int(cluster_labels[idx]),
            "profile_67d": speaker_profiles[sid].tolist(),
            "gender": "unknown",
        }
    clusters_path = output_dir / "clusters.json"
    with open(clusters_path, "w") as f:
        json.dump(clusters, f, indent=2)
    print(f"\nSaved {clusters_path} ({clusters_path.stat().st_size / 1024:.1f} KB)")

    # cluster_centroids.json
    centroids_json = {}
    for key in centroids_67d:
        centroids_json[key] = {
            "centroid_67d": centroids_67d[key].tolist(),
            "threshold": thresholds[key],
        }
    centroids_path = output_dir / "cluster_centroids.json"
    with open(centroids_path, "w") as f:
        json.dump(centroids_json, f, indent=2)
    print(f"Saved {centroids_path} ({centroids_path.stat().st_size / 1024:.1f} KB)")

    # identity_floors.json
    floors_path = output_dir / "identity_floors.json"
    with open(floors_path, "w") as f:
        json.dump(identity_floors, f, indent=2)
    print(f"Saved {floors_path} ({floors_path.stat().st_size / 1024:.1f} KB)")


# %%
# ═══════════════════════════════════════════════════════════════════════
# CELL 3 — Orchestrator
# ═══════════════════════════════════════════════════════════════════════

t_total = time.time()

# ── Step 1: Discover speakers ──
print("═══ Step 1: Discovering speakers ═══")
speakers = discover_speakers(PATHS)
total_clips = sum(len(v) for v in speakers.values())
print(f"Found {len(speakers)} speakers, {total_clips} total clips")
for ds in ["vctk", "ljspeech", "daps"]:
    ds_speakers = {k: v for k, v in speakers.items()
                   if (ds == "vctk" and k.startswith("p")) or
                      (ds == "ljspeech" and k.startswith("LJ")) or
                      (ds == "daps" and not k.startswith("p") and not k.startswith("LJ"))}
    if ds_speakers:
        ds_clips = sum(len(v) for v in ds_speakers.values())
        print(f"  {ds}: {len(ds_speakers)} speakers, {ds_clips} clips")

# ── Step 2: Compute speaker profiles ──
print("\n═══ Step 2: Computing speaker profiles ═══")
profiles = compute_speaker_profiles(speakers, SAMPLES_PER_SPEAKER)
print(f"\nComputed {len(profiles)} speaker profiles (67D each)")

# ── Step 3: Clustering ──
print("\n═══ Step 3: Clustering ═══")
labels, centroids_67d, thresholds, speaker_ids, X, X_scaled = \
    run_clustering(profiles, N_CLUSTERS)

# ── Step 4: t-SNE visualization ──
print("\n═══ Step 4: t-SNE visualization ═══")
perplexity = min(30, len(speaker_ids) - 1)
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
X_2d = tsne.fit_transform(X_scaled)

fig, ax = plt.subplots(1, 1, figsize=(12, 8))
palette = sns.color_palette("tab10", N_CLUSTERS)
for c in range(N_CLUSTERS):
    mask = labels == c
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=[palette[c]], label=f"Cluster {c}",
               s=60, alpha=0.7, edgecolors="white", linewidth=0.5)

# Annotate speakers
for i, sid in enumerate(speaker_ids):
    ax.annotate(sid, (X_2d[i, 0], X_2d[i, 1]),
                fontsize=7, alpha=0.6, ha="center", va="bottom")

ax.set_title("t-SNE of 67D Speaker Profiles (colored by cluster)")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

tsne_path = CLUSTER_DATA_DIR / "tsne_clusters.png"
fig.savefig(tsne_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved {tsne_path}")

# ── Step 5: Identity floors ──
print("\n═══ Step 5: Identity floors ═══")
floors = compute_identity_floors(profiles, labels, speaker_ids)

# ── Step 6: Validation ──
validate_results(labels, floors, centroids_67d, thresholds)

# ── Step 7: Save outputs ──
print("\n═══ Step 7: Saving outputs ═══")
save_outputs(profiles, labels, centroids_67d, thresholds, floors, speaker_ids, CLUSTER_DATA_DIR)

elapsed = time.time() - t_total
print(f"\n{'='*60}")
print(f"  PHASE 1 COMPLETE — {elapsed/60:.1f} min")
print(f"  Output: {CLUSTER_DATA_DIR}")
print(f"{'='*60}")
