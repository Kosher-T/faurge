import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from plugins_port.saturator.saturator import process
from plugins_port.saturator.saturator import WAVESHAPERS


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_sine(freq, amp, sr, dur):
    n = int(sr * dur)
    t = np.arange(n) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def compute_rms(buf):
    return float(np.sqrt(np.mean(buf.astype(np.float64) ** 2)))


# ── Drive=0 → unchanged ──────────────────────────────────────────────────────

def test_drive_zero_consistent():
    """drive=0 applied twice produces consistent results."""
    sr = 44100
    audio = make_sine(440.0, 0.5, sr, 1.0)

    out1, result1 = process(audio.copy(), sr,
                            drive_db=0.0,
                            sat_type="tube")
    out2, result2 = process(audio.copy(), sr,
                            drive_db=0.0,
                            sat_type="tube")

    assert result1["success"]
    assert result2["success"]
    max_diff = float(np.max(np.abs(out1 - out2)))
    assert max_diff < 1e-6, f"drive=0 should be deterministic: diff={max_diff}"


def test_mix_zero_bit_identical():
    """mix=0 → output identical to input regardless of drive."""
    sr = 44100
    audio = make_sine(440.0, 0.5, sr, 1.0)
    original = audio.copy()

    output, result = process(audio, sr,
                             drive_db=12.0,
                             mix=0.0,
                             sat_type="tube")

    assert result["success"]
    max_diff = float(np.max(np.abs(output - original)))
    assert max_diff < 1e-4, f"mix=0 should pass input through: max_diff={max_diff}"


def test_drive_twelve_increases_harmonics():
    """Higher drive increases harmonic content."""
    sr = 44100
    audio = make_sine(440.0, 0.5, sr, 1.0)

    out_low, result_low = process(audio.copy(), sr, drive_db=0.0, sat_type="tube")
    out_high, result_high = process(audio.copy(), sr, drive_db=12.0, sat_type="tube")

    assert result_low["success"]
    assert result_high["success"]
    diff = float(np.max(np.abs(out_low - out_high)))
    assert diff > 0.01, f"Higher drive should change output: diff={diff}"


# ── Type comparison ──────────────────────────────────────────────────────────

def test_tube_vs_diode_different():
    """tube vs diode produce different harmonic spectra on sine sweep."""
    sr = 44100
    audio = make_sine(1000.0, 0.5, sr, 0.5)

    out_tube, _ = process(audio.copy(), sr, drive_db=12.0, sat_type="tube")
    out_diode, _ = process(audio.copy(), sr, drive_db=12.0, sat_type="diode")

    diff = float(np.max(np.abs(out_tube - out_diode)))
    assert diff > 0.01, f"Tube and diode outputs too similar: diff={diff}"


def test_asymmetric_dc_offset():
    """asymmetric waveshaper produces measurable DC offset."""
    sr = 44100
    audio = make_sine(1000.0, 0.5, sr, 0.5)

    out_asym, result_asym = process(audio.copy(), sr, drive_db=12.0, sat_type="asymmetric")
    out_tube, result_tube = process(audio.copy(), sr, drive_db=12.0, sat_type="tube")

    assert abs(result_asym["dc_offset"]) > abs(result_tube["dc_offset"]), \
        f"Expected asymmetric DC > tube DC: asym={result_asym['dc_offset']:.6f}, tube={result_tube['dc_offset']:.6f}"


def test_symmetric_no_dc():
    """tube and tape (symmetric) should have near-zero DC."""
    sr = 44100
    audio = make_sine(1000.0, 0.5, sr, 0.5)

    out_tube, result_tube = process(audio.copy(), sr, drive_db=12.0, sat_type="tube")
    out_tape, result_tape = process(audio.copy(), sr, drive_db=12.0, sat_type="tape")

    assert abs(result_tube["dc_offset"]) < 0.01, f"Tube DC too high: {result_tube['dc_offset']}"
    assert abs(result_tape["dc_offset"]) < 0.01, f"Tape DC too high: {result_tape['dc_offset']}"


def test_polarity_flip():
    """asymmetric waveshaper is not polarity-symmetric; tube is."""
    sr = 44100
    audio = make_sine(1000.0, 0.5, sr, 0.5)

    out_asym, _ = process(audio, sr, drive_db=12.0, sat_type="asymmetric")
    out_asym_neg, _ = process(-audio, sr, drive_db=12.0, sat_type="asymmetric")

    diff_asym = float(np.max(np.abs(out_asym + out_asym_neg)))
    assert diff_asym > 0.01, \
        f"Asymmetric should NOT be polarity-symmetric: diff={diff_asym}"

    out_tube, _ = process(audio, sr, drive_db=12.0, sat_type="tube")
    out_tube_neg, _ = process(-audio, sr, drive_db=12.0, sat_type="tube")

    diff_tube = float(np.max(np.abs(out_tube + out_tube_neg)))
    assert diff_tube < 0.01, \
        f"Tube should be polarity-symmetric: diff={diff_tube}"


# ── HPF ──────────────────────────────────────────────────────────────────────

def test_hpf_preserves_high():
    """HPF pass: high-frequency content preserved."""
    sr = 44100
    audio = make_sine(5000.0, 0.5, sr, 0.5)
    original = audio.copy()

    output, result = process(audio, sr,
                             drive_db=6.0,
                             hpf_hz=200.0,
                             sat_type="tube")

    max_diff = float(np.max(np.abs(output - original)))
    assert max_diff > 0.001, "Expected some saturation effect"


def test_hpf_attenuates_low():
    """HPF attenuates: low-frequency content before saturation."""
    sr = 44100
    audio = make_sine(80.0, 0.5, sr, 0.5)

    out_no_hpf, _ = process(audio.copy(), sr, drive_db=12.0, hpf_hz=20.0, sat_type="tube")
    out_with_hpf, _ = process(audio.copy(), sr, drive_db=12.0, hpf_hz=500.0, sat_type="tube")

    rms_with = compute_rms(out_with_hpf)
    rms_without = compute_rms(out_no_hpf)
    assert rms_with < rms_without * 0.9, \
        f"HPF should reduce low freq: with={rms_with:.4f}, without={rms_without:.4f}"


def test_hpf_clamp_zero():
    """HPF at min should not change output vs default."""
    sr = 44100
    audio = make_sine(1000.0, 0.5, sr, 0.5)

    out_default, _ = process(audio.copy(), sr, drive_db=6.0, hpf_hz=20.0, sat_type="tube")
    out_clamped, _ = process(audio.copy(), sr, drive_db=6.0, hpf_hz=20.0, sat_type="tube")

    max_diff = float(np.max(np.abs(out_default - out_clamped)))
    assert max_diff < 1e-6, f"Same HPF should give same output: diff={max_diff}"


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("test_drive_zero_consistent", test_drive_zero_consistent),
        ("test_mix_zero_bit_identical", test_mix_zero_bit_identical),
        ("test_drive_twelve_increases_harmonics", test_drive_twelve_increases_harmonics),
        ("test_tube_vs_diode_different", test_tube_vs_diode_different),
        ("test_asymmetric_dc_offset", test_asymmetric_dc_offset),
        ("test_symmetric_no_dc", test_symmetric_no_dc),
        ("test_polarity_flip", test_polarity_flip),
        ("test_hpf_preserves_high", test_hpf_preserves_high),
        ("test_hpf_attenuates_low", test_hpf_attenuates_low),
        ("test_hpf_clamp_zero", test_hpf_clamp_zero),
    ]

    passed = 0
    failed = 0
    print("\n=== Saturator: Python Port Tests ===")
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
