from dataclasses import dataclass

# ══════════════════════════════════════════════════════════════════════════════
# Import portable plugins (Kaggle top-level modules)
# ══════════════════════════════════════════════════════════════════════════════

PLUGIN_BASE = Path('/kaggle/usr/lib/notebooks/itorousa')
sys.path.insert(0, str(PLUGIN_BASE))

import equalizer
import gain1

PLUGIN_MODULES = {
    'eq': equalizer,
    'gain': gain1,
}

print("Loaded plugins:", list(PLUGIN_MODULES.keys()))

# ══════════════════════════════════════════════════════════════════════════════
# ActionUnnormalizer — decode [-1,1] → real param ranges
# ══════════════════════════════════════════════════════════════════════════════
# Reuse the same ParamRange definitions from Phase 4
# (copied here for notebook self-containment)

@dataclass(frozen=True)
class ParamRange:
    name: str
    low: float
    high: float
    log: bool = False

EQ_PARAM_RANGES: List[ParamRange] = []
for _b in range(31):
    EQ_PARAM_RANGES.extend([
        ParamRange(f"eq_band{_b+1}_freq",        20.0,   20_000.0, log=True),
        ParamRange(f"eq_band{_b+1}_gain",       -24.0,      24.0),
        ParamRange(f"eq_band{_b+1}_q",            0.1,       10.0),
        ParamRange(f"eq_band{_b+1}_filter_type",  0.0,        2.0),
    ])

GAIN_PARAM_RANGES = [
    ParamRange("gain_db",           -12.0,     12.0),
]

ALL_PARAM_RANGES: List[ParamRange] = (
    EQ_PARAM_RANGES + GAIN_PARAM_RANGES
)

# Pre-compute bounds as numpy arrays for vectorized decode
PARAM_LOWS = np.array([pr.low for pr in ALL_PARAM_RANGES], dtype=np.float32)
PARAM_HIGHS = np.array([pr.high for pr in ALL_PARAM_RANGES], dtype=np.float32)
PARAM_IS_LOG = np.array([pr.log for pr in ALL_PARAM_RANGES], dtype=bool)

# Categorical indices
CATEGORICAL_INDICES: Dict[str, List[int]] = {
    "eq_filter_type": list(range(3, 124, 4)),
}
CAT_SET: set = set()
for indices in CATEGORICAL_INDICES.values():
    CAT_SET.update(indices)

# Plugin slices
PLUGIN_SLICES: Dict[str, Tuple[int, int]] = {}
_offset = 0
for _name, _count in [
    ("eq", 31 * 4), ("gain", 1),
]:
    PLUGIN_SLICES[_name] = (_offset, _offset + _count)
    _offset += _count

print(f"Param ranges: {len(ALL_PARAM_RANGES)}, Plugin slices: {PLUGIN_SLICES}")


def decode_action(action: np.ndarray) -> Dict[str, dict]:
    """
    Decode a single 125D action in [-1,1] → per-plugin config dicts.

    Args:
        action: (125,) array in [-1, 1]
    Returns:
        dict with keys: eq, gain
    """
    assert action.shape == (OUTPUT_DIM,), f"Expected ({OUTPUT_DIM},), got {action.shape}"

    # Linear scale (continuous, non-log, non-categorical)
    vals = np.zeros_like(action)
    cont_mask = np.array([i not in CAT_SET and not PARAM_IS_LOG[i] for i in range(OUTPUT_DIM)])
    vals[cont_mask] = (action[cont_mask] + 1.0) * 0.5 * (PARAM_HIGHS[cont_mask] - PARAM_LOWS[cont_mask]) + PARAM_LOWS[cont_mask]

    # Log scale
    log_mask = np.array([i not in CAT_SET and PARAM_IS_LOG[i] for i in range(OUTPUT_DIM)])
    log_lows = np.log(np.maximum(PARAM_LOWS[log_mask], 1e-8))
    log_highs = np.log(np.maximum(PARAM_HIGHS[log_mask], 1e-8))
    vals[log_mask] = np.exp((action[log_mask] + 1.0) * 0.5 * (log_highs - log_lows) + log_lows)

    # Categorical: nearest integer bin
    cat_indices = sorted(CAT_SET)
    cat_arr = np.array(cat_indices)
    cat_vals = np.round((action[cat_arr] + 1.0) * 0.5 * (PARAM_HIGHS[cat_arr] - PARAM_LOWS[cat_arr]) + PARAM_LOWS[cat_arr])
    cat_vals = np.clip(cat_vals, PARAM_LOWS[cat_arr], PARAM_HIGHS[cat_arr])
    vals[cat_arr] = cat_vals

    # Map to named params
    params = {pr.name: vals[i] for i, pr in enumerate(ALL_PARAM_RANGES)}

    # Build per-plugin config dicts
    _FTYPES = ["peak", "low_shelf", "high_shelf"]

    # EQ: 31 bands
    eq_bands = []
    for b in range(31):
        ftype_idx = int(round(params[f"eq_band{b+1}_filter_type"]))
        ftype_idx = max(0, min(2, ftype_idx))
        eq_bands.append({
            "freq_hz": float(params[f"eq_band{b+1}_freq"]),
            "gain_db": float(params[f"eq_band{b+1}_gain"]),
            "q": float(params[f"eq_band{b+1}_q"]),
            "filter_type": _FTYPES[ftype_idx],
            "stereo_skew_db": 0.0,
            "dynamic_depth": 0.0,
        })

    # Gain
    g = {
        "gain_db": float(params["gain_db"]),
    }

    return {
        "eq": eq_bands,
        "gain": g,
    }


def apply_plugins(audio: np.ndarray, sr: int, plugin_dicts: dict) -> np.ndarray:
    """
    Apply the 2-plugin cascade (EQ + Gain) in order to audio.

    Args:
        audio: (N,) or (N, C) float32
        sr: sample rate
        plugin_dicts: dict from decode_action()
    Returns:
        processed audio, same shape as input
    """
    result = audio.copy()

    # 1. EQ (module: equalizer)
    result, _ = equalizer.process(result, sr, bands=plugin_dicts['eq'])

    # 2. Gain (module: gain1)
    result, _ = gain1.process(result, sr=sr, **plugin_dicts['gain'])

    return result


# ══════════════════════════════════════════════════════════════════════════════
# 67D Metrics Extraction (same as Phase 3)
# ══════════════════════════════════════════════════════════════════════════════

import soundfile as sf
import librosa

BARK_N_BANDS = 64
BARK_LOW_HZ = 20.0
BARK_HIGH_HZ = 20000.0
FFT_SIZE = 4096
HOP_SIZE = 1024


def bark_edges(n_bands: int, low_hz: float, high_hz: float) -> np.ndarray:
    """Bark-scale frequency edges."""
    low_bark = 13 * np.arctan(0.00076 * low_hz) + 3.5 * np.arctan((low_hz / 7500.0) ** 2)
    high_bark = 13 * np.arctan(0.00076 * high_hz) + 3.5 * np.arctan((high_hz / 7500.0) ** 2)
    edges_bark = np.linspace(low_bark, high_bark, n_bands + 1)
    # Convert back to Hz
    # Inverse of bark = 13*atan(0.00076*f) + 3.5*atan((f/7500)^2)
    # Use bisection for inverse
    edges_hz = np.zeros_like(edges_bark)
    for i, b in enumerate(edges_bark):
        lo, hi = 0.0, 100000.0
        for _ in range(50):
            mid = (lo + hi) / 2
            bark_mid = 13 * np.arctan(0.00076 * mid) + 3.5 * np.arctan((mid / 7500.0) ** 2)
            if bark_mid < b:
                lo = mid
            else:
                hi = mid
        edges_hz[i] = (lo + hi) / 2
    return edges_hz


BARK_EDGES = bark_edges(BARK_N_BANDS, BARK_LOW_HZ, BARK_HIGH_HZ)


def compute_ltas_64(audio: np.ndarray, sr: int) -> np.ndarray:
    """Compute Long-Term Average Spectrum in 64 Bark bands (dB)."""
    S = np.abs(np.fft.rfft(audio, n=FFT_SIZE)) ** 2
    freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / sr)
    ltas = np.zeros(BARK_N_BANDS)
    for i in range(BARK_N_BANDS):
        mask = (freqs >= BARK_EDGES[i]) & (freqs < BARK_EDGES[i + 1])
        if mask.any():
            ltas[i] = 10 * np.log10(S[mask].mean() + 1e-10)
        else:
            ltas[i] = -100.0
    return ltas


def compute_lufs(audio: np.ndarray, sr: int) -> float:
    """Simplified integrated LUFS (K-weighting approximation)."""
    # K-weighting: pre-filter + RLU
    # Simplified: just use A-weighting approximation
    S = np.abs(np.fft.rfft(audio, n=FFT_SIZE)) ** 2
    freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / sr)
    # K-weighting approximation (simplified)
    weights = np.ones_like(freqs)
    weights[freqs < 100] = 0.5
    weights[freqs > 10000] = 0.8
    loudness = np.sum(S * weights) / len(audio)
    if loudness < 1e-10:
        return -70.0
    return -0.691 + 10 * np.log10(loudness + 1e-10)


def compute_crest_db(audio: np.ndarray) -> float:
    """Crest factor in dB (peak - RMS)."""
    peak = np.max(np.abs(audio))
    rms = np.sqrt(np.mean(audio ** 2) + 1e-10)
    if rms < 1e-10:
        return 0.0
    return 20 * np.log10(peak / rms)


def compute_zcr(audio: np.ndarray) -> float:
    """Zero-crossing rate (normalized)."""
    signs = np.sign(audio)
    signs[signs == 0] = 1
    crossings = np.sum(np.abs(np.diff(signs))) / (2 * len(audio))
    return float(crossings)


def extract_metrics_67d(audio: np.ndarray) -> np.ndarray:
    """Extract 67D feature vector: LTAS(64) + LUFS(1) + Crest(1) + ZCR(1)."""
    ltas = compute_ltas_64(audio, SR)
    lufs = np.array([compute_lufs(audio, SR)])
    crest = np.array([compute_crest_db(audio)])
    zcr = np.array([compute_zcr(audio)])
    return np.concatenate([ltas, lufs, crest, zcr]).astype(np.float32)


def compute_reward(mse, floor, initial_mse):
    """Reward based on absolute MSE on a fixed linear scale.

        mse == 1                → +1.0  (near-perfect restoration)
        mse == REWARD_MSE_MAX   → -1.0  (very wrong)
    """
    reward = 1.0 - (mse - 1.0) * 2.0 / (REWARD_MSE_MAX - 1.0)
    return float(np.clip(reward, -1.0, 1.0))


print("Functions defined: decode_action, apply_plugins, extract_metrics_67d, compute_reward")
