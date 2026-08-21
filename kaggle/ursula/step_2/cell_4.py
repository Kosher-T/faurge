# ══════════════════════════════════════════════════════════════════════════════
# Step 1be — Degradation Function (Gain Only)
# ══════════════════════════════════════════════════════════════════════════════

import sys
sys.path.insert(0, str(PLUGIN_DIR / 'gain1'))
import gain1 as gain_plugin

def degrade_with_gain(audio, sr=SR):
    """Apply random gain. Return degraded audio."""
    gain_db = np.random.uniform(*GAIN_RANGE_DB)
    degraded, _ = gain_plugin.process(audio, sr=sr, gain_db=float(gain_db))
    return degraded

print(f"Degradation: random gain {GAIN_RANGE_DB}")
