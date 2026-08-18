# ══════════════════════════════════════════════════════════════════════════════
# Step 1ba — Generate Degradations (Fixed EQ, Gain-Only)
# ══════════════════════════════════════════════════════════════════════════════
# Fixed EQ band (1kHz peak), random gain. Label = gain_db (scalar).

import sys
sys.path.insert(0, str(PLUGIN_DIR / 'equalizer'))
import equalizer as eq_plugin

def degrade_with_eq(audio, sr=SR):
    """Apply fixed EQ with random gain. Return (degraded_audio, gain_db)."""
    gain_db = np.random.uniform(*EQ_GAIN_RANGE_DB)

    degraded, _ = eq_plugin.process(
        audio, sr=sr,
        bands=[{
            "freq_hz": EQ_FIXED_FREQ_HZ,
            "gain_db": float(gain_db),
            "q": EQ_FIXED_Q,
            "filter_type": "peak"
        }]
    )

    return degraded, np.float32(gain_db)

print(f"Degradation defined: fixed EQ {EQ_FIXED_FREQ_HZ}Hz, Q={EQ_FIXED_Q}, random gain {EQ_GAIN_RANGE_DB}")
