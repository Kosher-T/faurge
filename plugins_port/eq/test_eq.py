import os
import sys

if __name__ == "__main__":
    # Add parent directory to path so plugins_port is importable
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    
    from plugins_port.eq.test.test_sweep import (
        test_single_peak_band_boosts_at_center_freq,
        test_single_peak_band_cuts_at_center_freq,
        test_peak_band_does_not_affect_distant_freq,
        test_multiple_bands_accumulate_gain
    )
    from plugins_port.eq.test.test_notch import (
        test_notch_creates_null_at_1kHz,
        test_notch_leaves_distant_freq_untouched,
        test_notch_multiple_freqs_all_notched
    )
    from plugins_port.eq.test.test_skew import (
        test_stereo_skew_produces_lr_difference,
        test_stereo_skew_difference_only_in_band_range
    )
    from plugins_port.eq.test.test_dynamic import (
        test_dynamic_eq_depth_zero_matches_static,
        test_dynamic_depth_one_varies_with_envelope
    )
    
    tests = [
        ("test_single_peak_band_boosts_at_center_freq", test_single_peak_band_boosts_at_center_freq),
        ("test_single_peak_band_cuts_at_center_freq", test_single_peak_band_cuts_at_center_freq),
        ("test_peak_band_does_not_affect_distant_freq", test_peak_band_does_not_affect_distant_freq),
        ("test_multiple_bands_accumulate_gain", test_multiple_bands_accumulate_gain),
        ("test_notch_creates_null_at_1kHz", test_notch_creates_null_at_1kHz),
        ("test_notch_leaves_distant_freq_untouched", test_notch_leaves_distant_freq_untouched),
        ("test_notch_multiple_freqs_all_notched", test_notch_multiple_freqs_all_notched),
        ("test_stereo_skew_produces_lr_difference", test_stereo_skew_produces_lr_difference),
        ("test_stereo_skew_difference_only_in_band_range", test_stereo_skew_difference_only_in_band_range),
        ("test_dynamic_eq_depth_zero_matches_static", test_dynamic_eq_depth_zero_matches_static),
        ("test_dynamic_depth_one_varies_with_envelope", test_dynamic_depth_one_varies_with_envelope),
    ]
    
    passed = 0
    failed = 0
    print("\n=== EQ: Python Port Tests ===")
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
