# ══════════════════════════════════════════════════════════════════════════════
# Step 1bc — Degradation Function
# ══════════════════════════════════════════════════════════════════════════════

import sys
sys.path.insert(0, str(PLUGIN_DIR / 'equalizer'))
import equalizer as eq_plugin

def degrade_with_eq(audio, freq_hz, sr=SR):
    """Apply EQ at given frequency with random gain. Return (degraded_audio, gain_db)."""
    gain_db = np.random.uniform(*EQ_GAIN_RANGE_DB)
    degraded, _ = eq_plugin.process(
        audio, sr=sr,
        bands=[{
            "freq_hz": float(freq_hz),
            "gain_db": float(gain_db),
            "q": EQ_FIXED_Q,
            "filter_type": "peak"
        }]
    )
    return degraded, np.float32(gain_db)

print(f"Degradation: random gain {EQ_GAIN_RANGE_DB}, fixed Q={EQ_FIXED_Q}")
