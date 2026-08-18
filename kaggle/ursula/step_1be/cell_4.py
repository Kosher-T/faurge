# ══════════════════════════════════════════════════════════════════════════════
# Step 1be — Degradation Functions
# ══════════════════════════════════════════════════════════════════════════════
# Both phases use EQ degradation — gain-only doesn't change Tier 1 features
# (spectral shape), so the model can't learn them. EQ forces the model to
# understand how spectral features change with degradation.

import sys
sys.path.insert(0, str(PLUGIN_DIR / 'gain1'))
import gain1 as gain_plugin
sys.path.insert(0, str(PLUGIN_DIR / 'equalizer'))
import equalizer as eq_plugin

def degrade_eq_only(audio, freq_pool, sr=SR):
    """Apply 1 EQ band (freq from pool). No global gain.
    Returns (degraded_audio, freq_hz, eq_gain_db).
    Used for Phase 1 metric literacy — model learns what EQ changes look like."""
    freq_hz = float(np.random.choice(freq_pool))
    eq_gain_db = np.random.uniform(*EQ_GAIN_RANGE_DB)

    degraded, _ = eq_plugin.process(
        audio, sr=sr,
        bands=[{
            "freq_hz": freq_hz,
            "gain_db": float(eq_gain_db),
            "q": EQ_FIXED_Q,
            "filter_type": "peak",
        }]
    )
    degraded = np.clip(degraded, -1.0, 1.0)
    return degraded, freq_hz, eq_gain_db

def degrade_eq_gain(audio, freq_pool, sr=SR):
    """Apply 1 EQ band (freq from pool) + global gain.
    Returns (degraded_audio, targets_3d) where
    targets_3d = [freq_norm, eq_gain_norm, global_gain_norm]."""
    freq_hz = float(np.random.choice(freq_pool))
    eq_gain_db = np.random.uniform(*EQ_GAIN_RANGE_DB)
    global_gain_db = np.random.uniform(*GAIN_RANGE_DB)

    degraded, _ = eq_plugin.process(
        audio, sr=sr,
        bands=[{
            "freq_hz": freq_hz,
            "gain_db": float(eq_gain_db),
            "q": EQ_FIXED_Q,
            "filter_type": "peak",
        }]
    )
    degraded, _ = gain_plugin.process(degraded, sr=sr, gain_db=float(global_gain_db))
    degraded = np.clip(degraded, -1.0, 1.0)

    targets = np.array([
        normalize_freq(freq_hz),
        normalize_eq_gain(eq_gain_db),
        normalize_global_gain(global_gain_db),
    ], dtype=np.float32)

    return degraded, targets

print("Degradation functions loaded:")
print(f"  Phase 1: EQ only (1 band, Q={EQ_FIXED_Q}, gain {EQ_GAIN_RANGE_DB})")
print(f"  Phase 2: EQ + global gain (1 band + gain {GAIN_RANGE_DB})")
