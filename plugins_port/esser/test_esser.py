import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from plugins_port.esser.esser import process
from plugins_port.eq.eq import process as eq_process, DEFAULT_BAND


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_sine(freq, amp, sr, dur):
    n = int(sr * dur)
    t = np.arange(n) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def compute_rms(buf):
    return float(np.sqrt(np.mean(buf.astype(np.float64) ** 2)))


# ── Test: Gated 7kHz tone ────────────────────────────────────────────────────

def test_gated_tone():
    """White noise with gated 7kHz tone — esser attenuates only when tone is present."""
    sr = 48000
    dur = 1.0
    n = int(sr * dur)
    t = np.arange(n) / sr

    np.random.seed(42)
    white = 0.05 * np.random.randn(n).astype(np.float32)

    # Add 7kHz tone in the middle third only
    tone_start = n // 3
    tone_end = 2 * n // 3
    tone = np.zeros(n, dtype=np.float32)
    tone[tone_start:tone_end] = 0.5 * np.sin(2.0 * np.pi * 7000.0 * t[tone_start:tone_end])

    audio = white + tone

    output, result = process(audio, sr,
                             center_freq_hz=7000.0,
                             threshold_db=-20.0,
                             ratio=10.0,
                             bandwidth_hz=1500.0,
                             attack_ms=1.0,
                             release_ms=50.0)

    assert result["success"]
    assert result["sibilant_frames"] > 0, "Expected sibilant frames detected"

    # RMS in tone region should be lower after processing
    rms_orig_tone = compute_rms(audio[tone_start:tone_end])
    rms_proc_tone = compute_rms(output[tone_start:tone_end])
    assert rms_proc_tone < rms_orig_tone * 0.8, \
        f"Expected esser to attenuate tone region: orig={rms_orig_tone:.4f}, proc={rms_proc_tone:.4f}"

    # RMS in non-tone regions should be roughly unchanged
    rms_orig_quiet = compute_rms(audio[:tone_start])
    rms_proc_quiet = compute_rms(output[:tone_start])
    assert abs(rms_orig_quiet - rms_proc_quiet) < rms_orig_quiet * 0.5, \
        f"Non-tone region changed too much: orig={rms_orig_quiet:.4f}, proc={rms_proc_quiet:.4f}"


# ── Test: ratio=1 → no change ───────────────────────────────────────────────

def test_ratio_one_no_change():
    """ratio=1 → no change regardless of content."""
    sr = 48000
    audio = make_sine(7000.0, 0.5, sr, 0.5)
    original = audio.copy()

    output, result = process(audio, sr,
                             center_freq_hz=7000.0,
                             threshold_db=-60.0,
                             ratio=1.0,
                             bandwidth_hz=1500.0)

    assert result["success"]
    max_diff = float(np.max(np.abs(output - original)))
    assert max_diff < 1e-4, f"max_diff={max_diff}"


# ── Test: Static EQ cannot replicate esser's threshold-gated behavior ────────

def test_vs_eq_threshold_gated():
    """Static EQ can NOT replicate esser's threshold-gated behavior."""
    sr = 48000
    dur = 1.0
    n = int(sr * dur)
    t = np.arange(n) / sr

    # Quiet signal (below threshold) — esser should NOT act
    quiet = make_sine(7000.0, 0.01, sr, dur)
    # Loud signal (above threshold) — esser SHOULD act
    loud = make_sine(7000.0, 0.5, sr, dur)

    # Process both through esser
    out_quiet, _ = process(quiet, sr,
                           center_freq_hz=7000.0,
                           threshold_db=-20.0,
                           ratio=10.0,
                           bandwidth_hz=1500.0)
    out_loud, _ = process(loud, sr,
                          center_freq_hz=7000.0,
                          threshold_db=-20.0,
                          ratio=10.0,
                          bandwidth_hz=1500.0)

    # Quiet: esser should leave it nearly unchanged
    quiet_ratio = compute_rms(out_quiet) / max(compute_rms(quiet), 1e-30)

    # Loud: esser should attenuate significantly
    loud_ratio = compute_rms(out_loud) / max(compute_rms(loud), 1e-30)

    # The key test: quiet_ratio should be near 1.0, loud_ratio should be << 1.0
    assert quiet_ratio > 0.9, f"Quiet ratio should be near 1.0, got {quiet_ratio:.4f}"
    assert loud_ratio < 0.7, f"Loud ratio should be << 1.0, got {loud_ratio:.4f}"
    assert quiet_ratio > loud_ratio, \
        f"Threshold gating failed: quiet_ratio={quiet_ratio:.4f} <= loud_ratio={loud_ratio:.4f}"


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("test_gated_tone", test_gated_tone),
        ("test_ratio_one_no_change", test_ratio_one_no_change),
        ("test_vs_eq_threshold_gated", test_vs_eq_threshold_gated),
    ]

    passed = 0
    failed = 0
    print("\n=== Esser: Python Port Tests ===")
    for name, fn in tests:
        print(f"  [RUN]  {name}")
        try:
            fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n  Results: {passed} passed, {failed} failed\n")
    sys.exit(1 if failed > 0 else 0)
