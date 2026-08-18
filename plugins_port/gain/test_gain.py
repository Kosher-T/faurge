import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from plugins_port.gain.gain import process, _measure_lufs


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_sine(freq, amp, sr, dur):
    n = int(sr * dur)
    t = np.arange(n) / sr
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def make_stereo(left, right):
    n = max(len(left), len(right))
    stereo = np.zeros((n, 2), dtype=np.float32)
    stereo[:len(left), 0] = left
    stereo[:len(right), 1] = right
    return stereo


def compute_rms_channel(buf, ch):
    return float(np.sqrt(np.mean(buf[:, ch].astype(np.float64) ** 2)))


def compute_energy_ratio(buf):
    left = float(np.mean(buf[:, 0].astype(np.float64) ** 2))
    right = float(np.mean(buf[:, 1].astype(np.float64) ** 2))
    if left + right < 1e-30:
        return 0.5
    return left / (left + right)


# ── Identity test ────────────────────────────────────────────────────────────

def test_identity_mono():
    """gain=0, balance=0 → bit-identical output for mono."""
    sr = 44100
    audio = make_sine(440.0, 0.5, sr, 1.0)
    original = audio.copy()

    output, result = process(audio, sr, gain_db=0.0, stereo_balance=0.0)

    assert result["success"]
    max_diff = float(np.max(np.abs(output - original)))
    assert max_diff < 1e-6, f"mono identity: max_diff={max_diff}"


def test_identity_stereo():
    """gain=0, balance=0 → bit-identical output for stereo."""
    sr = 44100
    left = make_sine(440.0, 0.5, sr, 1.0)
    right = make_sine(660.0, 0.3, sr, 1.0)
    audio = make_stereo(left, right)
    original = audio.copy()

    output, result = process(audio, sr, gain_db=0.0, stereo_balance=0.0)

    assert result["success"]
    max_diff = float(np.max(np.abs(output - original)))
    assert max_diff < 1e-6, f"stereo identity: max_diff={max_diff}"


def test_identity_no_sr():
    """process works without sr (sr not required for gain-only)."""
    sr = 44100
    audio = make_sine(440.0, 0.5, sr, 1.0)
    original = audio.copy()

    output, result = process(audio, gain_db=0.0, stereo_balance=0.0)

    assert result["success"]
    max_diff = float(np.max(np.abs(output - original)))
    assert max_diff < 1e-6, f"no-sr identity: max_diff={max_diff}"


def test_identity_gain_and_balance():
    """gain=0, balance=0, sr provided → bit-identical."""
    sr = 44100
    audio = make_sine(440.0, 0.5, sr, 1.0)
    original = audio.copy()

    output, result = process(audio, sr, gain_db=0.0, stereo_balance=0.0)

    assert result["success"]
    assert result["peak_change_db"] == 0.0
    assert result["applied_balance"] == 0.0
    max_diff = float(np.max(np.abs(output - original)))
    assert max_diff < 1e-6, f"gain+balance identity: max_diff={max_diff}"


def test_identity_nonzero_sr():
    """gain=0, balance=0, non-standard sr → bit-identical."""
    sr = 96000
    audio = make_sine(440.0, 0.5, sr, 1.0)
    original = audio.copy()

    output, result = process(audio, sr, gain_db=0.0, stereo_balance=0.0)

    assert result["success"]
    max_diff = float(np.max(np.abs(output - original)))
    assert max_diff < 1e-6, f"nonstandard sr identity: max_diff={max_diff}"


# ── Balance tests ────────────────────────────────────────────────────────────

def test_balance_full_left():
    """balance=-1 → all energy in left channel only."""
    sr = 44100
    left = make_sine(440.0, 0.5, sr, 1.0)
    right = make_sine(440.0, 0.5, sr, 1.0)
    audio = make_stereo(left, right)

    output, result = process(audio, sr, gain_db=0.0, stereo_balance=-1.0)

    assert result["success"]
    assert result["applied_balance"] == -1.0
    right_energy = float(np.mean(output[:, 1] ** 2))
    assert right_energy < 1e-10, f"Right channel should be silent: energy={right_energy}"
    left_energy = float(np.mean(output[:, 0] ** 2))
    assert left_energy > 0, "Left channel should have energy"


def test_balance_full_right():
    """balance=+1 → all energy in right channel only."""
    sr = 44100
    left = make_sine(440.0, 0.5, sr, 1.0)
    right = make_sine(440.0, 0.5, sr, 1.0)
    audio = make_stereo(left, right)

    output, result = process(audio, sr, gain_db=0.0, stereo_balance=1.0)

    assert result["success"]
    left_energy = float(np.mean(output[:, 0] ** 2))
    assert left_energy < 1e-10, f"Left channel should be silent: energy={left_energy}"
    right_energy = float(np.mean(output[:, 1] ** 2))
    assert right_energy > 0, "Right channel should have energy"


def test_balance_center():
    """balance=0 → equal energy in both channels."""
    sr = 44100
    left = make_sine(440.0, 0.5, sr, 1.0)
    right = make_sine(440.0, 0.5, sr, 1.0)
    audio = make_stereo(left, right)

    output, result = process(audio, sr, gain_db=0.0, stereo_balance=0.0)

    assert result["success"]
    ratio = compute_energy_ratio(output)
    assert abs(ratio - 0.5) < 0.01, f"Center balance ratio should be ~0.5: {ratio}"


def test_balance_partial_left():
    """balance=-0.5 → more energy in left, some right."""
    sr = 44100
    left = make_sine(440.0, 0.5, sr, 1.0)
    right = make_sine(440.0, 0.5, sr, 1.0)
    audio = make_stereo(left, right)

    output, result = process(audio, sr, gain_db=0.0, stereo_balance=-0.5)

    assert result["success"]
    ratio = compute_energy_ratio(output)
    assert ratio > 0.5, f"Left bias should have ratio > 0.5: {ratio}"


def test_balance_partial_right():
    """balance=+0.5 → more energy in right, some left."""
    sr = 44100
    left = make_sine(440.0, 0.5, sr, 1.0)
    right = make_sine(440.0, 0.5, sr, 1.0)
    audio = make_stereo(left, right)

    output, result = process(audio, sr, gain_db=0.0, stereo_balance=0.5)

    assert result["success"]
    ratio = compute_energy_ratio(output)
    assert ratio < 0.5, f"Right bias should have ratio < 0.5: {ratio}"


def test_balance_mono_noop():
    """Balance on mono (1D input) should be a no-op (balance == 0 internally)."""
    sr = 44100
    audio = make_sine(440.0, 0.5, sr, 1.0)
    original = audio.copy()

    output, result = process(audio, sr, gain_db=0.0, stereo_balance=-1.0)

    assert result["success"]
    max_diff = float(np.max(np.abs(output - original)))
    assert max_diff < 1e-6, f"mono balance should be no-op: max_diff={max_diff}"


# ── Gain tests ───────────────────────────────────────────────────────────────

def test_gain_positive_increases_peak():
    """Positive gain increases output peak."""
    sr = 44100
    audio = make_sine(440.0, 0.5, sr, 1.0)

    output, result = process(audio, sr, gain_db=6.0, stereo_balance=0.0)

    assert result["success"]
    assert result["peak_change_db"] > 0, "Positive gain should increase peak"
    assert result["output_peak_db"] > result["input_peak_db"]


def test_gain_negative_decreases_peak():
    """Negative gain decreases output peak."""
    sr = 44100
    audio = make_sine(440.0, 0.5, sr, 1.0)

    output, result = process(audio, sr, gain_db=-6.0, stereo_balance=0.0)

    assert result["success"]
    assert result["peak_change_db"] < 0, "Negative gain should decrease peak"
    assert result["output_peak_db"] < result["input_peak_db"]


# ── Clipping tests ───────────────────────────────────────────────────────────

def test_clipping_detected():
    """High gain that pushes signal past 0 dBFS sets clipping flag."""
    sr = 44100
    audio = make_sine(440.0, 0.9, sr, 1.0)

    output, result = process(audio, sr, gain_db=12.0, stereo_balance=0.0)

    assert result["success"]
    assert result["clipping"], "Expected clipping flag for high gain on loud signal"


def test_no_clipping_low_gain():
    """Low gain on quiet signal should not clip."""
    sr = 44100
    audio = make_sine(440.0, 0.1, sr, 1.0)

    output, result = process(audio, sr, gain_db=-12.0, stereo_balance=0.0)

    assert result["success"]
    assert not result["clipping"], "No clipping expected for low gain on quiet signal"


# ── LUFS tests ───────────────────────────────────────────────────────────────

def test_lufs_measurement_sine():
    """_measure_lufs returns a finite value for a sine tone."""
    sr = 44100
    audio = make_sine(440.0, 0.5, sr, 2.0)

    lufs = _measure_lufs(audio, sr)
    assert lufs > -120.0, f"LUFS should be measurable: {lufs}"
    assert lufs < 0.0, f"LUFS should be negative: {lufs}"


def test_lufs_measurement_silence():
    """_measure_lufs returns -120 for silence."""
    sr = 44100
    audio = np.zeros(int(sr * 2.0), dtype=np.float32)

    lufs = _measure_lufs(audio, sr)
    assert lufs <= -100.0, f"Silence LUFS should be very low: {lufs}"


def test_lufs_measurement_stereo():
    """_measure_lufs works on stereo audio."""
    sr = 44100
    left = make_sine(440.0, 0.5, sr, 2.0)
    right = make_sine(660.0, 0.3, sr, 2.0)
    audio = make_stereo(left, right)

    lufs = _measure_lufs(audio, sr)
    assert lufs > -120.0, f"Stereo LUFS should be measurable: {lufs}"


def test_lufs_in_report():
    """LUFS values are included in process result."""
    sr = 44100
    audio = make_sine(440.0, 0.5, sr, 2.0)

    output, result = process(audio, sr, gain_db=0.0, stereo_balance=0.0)

    assert result["success"]
    assert "input_lufs" in result
    assert "output_lufs" in result
    assert result["input_lufs"] > -120.0
    assert result["output_lufs"] > -120.0


# ── Runner ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        ("test_identity_mono", test_identity_mono),
        ("test_identity_stereo", test_identity_stereo),
        ("test_identity_no_sr", test_identity_no_sr),
        ("test_identity_gain_and_balance", test_identity_gain_and_balance),
        ("test_identity_nonzero_sr", test_identity_nonzero_sr),
        ("test_balance_full_left", test_balance_full_left),
        ("test_balance_full_right", test_balance_full_right),
        ("test_balance_center", test_balance_center),
        ("test_balance_partial_left", test_balance_partial_left),
        ("test_balance_partial_right", test_balance_partial_right),
        ("test_balance_mono_noop", test_balance_mono_noop),
        ("test_gain_positive_increases_peak", test_gain_positive_increases_peak),
        ("test_gain_negative_decreases_peak", test_gain_negative_decreases_peak),
        ("test_clipping_detected", test_clipping_detected),
        ("test_no_clipping_low_gain", test_no_clipping_low_gain),
        ("test_lufs_measurement_sine", test_lufs_measurement_sine),
        ("test_lufs_measurement_silence", test_lufs_measurement_silence),
        ("test_lufs_measurement_stereo", test_lufs_measurement_stereo),
        ("test_lufs_in_report", test_lufs_in_report),
    ]

    passed = 0
    failed = 0
    print("\n=== Gain: Python Port Tests ===")
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
