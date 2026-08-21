# ══════════════════════════════════════════════════════════════════════════════
# Step 2a — Degradation Function (EQ Only)
# ══════════════════════════════════════════════════════════════════════════════
# EQ degradation changes spectral shape — forces model to learn Tier 1 features.
# Random frequency (log-uniform 20-18k Hz), random gain (±12dB), fixed Q=1.0.

import sys
sys.path.insert(0, str(PLUGIN_DIR / 'equalizer'))
import equalizer as eq_plugin

def degrade_with_eq(audio, sr=SR):
    """Apply random EQ band. Return degraded audio."""
    freq_hz = float(np.exp(np.random.uniform(
        np.log(EQ_FREQ_MIN_HZ), np.log(EQ_FREQ_MAX_HZ))))
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
    return np.clip(degraded, -1.0, 1.0)

print(f"Degradation: EQ (freq {EQ_FREQ_MIN_HZ}-{EQ_FREQ_MAX_HZ} Hz, gain {EQ_GAIN_RANGE_DB}, Q={EQ_FIXED_Q})")
