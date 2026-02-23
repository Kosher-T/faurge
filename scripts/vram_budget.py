#!/usr/bin/env python3
"""
scripts/vram_budget.py
CLI VRAM / RAM budget audit for Faurge Phase 2.

Loads a model (or a synthetic tensor in --dry-run mode), runs a short
inference loop to profile peak memory including fragmentation, and halts
if the peak exceeds the configured safety ceiling.

Usage:
    python scripts/vram_budget.py --dry-run
    python scripts/vram_budget.py --model-path models/fabian.h5

Exit codes:
    0 — budget check passed
    1 — budget check failed or runtime error
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so we can import core.*
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.logging import get_logger
from core.memory_tracker import MemoryTracker
from core.vram_policy import (
    BudgetExceededError,
    enforce_budget,
    load_budget_policy,
)

log = get_logger("faurge.vram_budget")


# ==============================================================================
# --- Synthetic / Dry-Run Workload ---
# ==============================================================================

def _run_synthetic_workload(tracker: MemoryTracker, policy) -> None:
    """
    Allocate and iterate over a dummy tensor to simulate
    a model loading + inference cycle.  Good for CI and GPU-less systems.
    """
    import numpy as np

    log.info("--- Synthetic dry-run workload ---")

    with tracker.track_layer("synthetic_model_load"):
        # ~250 MB float32 tensor (65M floats × 4 bytes)
        dummy_weights = np.random.randn(8192, 8192).astype(np.float32)

    # Warmup
    for i in range(policy.warmup_steps):
        with tracker.track_layer(f"warmup_{i}"):
            _ = dummy_weights @ np.random.randn(8192, 1).astype(np.float32)

    # Measured inference steps
    for i in range(policy.inference_steps):
        with tracker.track_layer(f"inference_{i}"):
            _ = dummy_weights @ np.random.randn(8192, 1).astype(np.float32)
        tracker.snapshot(f"inference_step_{i}")

    # Cleanup
    del dummy_weights


# ==============================================================================
# --- Real Model Workload ---
# ==============================================================================

def _run_model_workload(
    model_path: str, tracker: MemoryTracker, policy
) -> None:
    """
    Load a real model from disk, run warmup + inference steps, and
    profile memory throughout.
    """
    import numpy as np

    log.info("--- Real model workload: %s ---", model_path)

    with tracker.track_layer("model_load"):
        # Attempt TensorFlow/Keras first
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(model_path)
            input_shape = model.input_shape
            # Build a dummy input matching the model's expected shape
            batch_shape = tuple(
                d if d is not None else 1 for d in input_shape
            )
            dummy_input = np.random.randn(*batch_shape).astype(np.float32)
        except Exception as e:
            log.error("Failed to load model at '%s': %s", model_path, e)
            raise

    tracker.snapshot("post_model_load")

    # Warmup
    for i in range(policy.warmup_steps):
        with tracker.track_layer(f"warmup_{i}"):
            _ = model.predict(dummy_input, verbose=0)

    # Measured inference
    for i in range(policy.inference_steps):
        with tracker.track_layer(f"inference_{i}"):
            _ = model.predict(dummy_input, verbose=0)
        tracker.snapshot(f"inference_step_{i}")


# ==============================================================================
# --- Report Printer ---
# ==============================================================================

def _print_report(report: dict, result) -> None:
    """Pretty-print the profiling results to stdout."""
    print("\n" + "=" * 60)
    print("  FAURGE MEMORY BUDGET AUDIT REPORT")
    print("=" * 60)
    print(f"  Mode:      {report['mode'].upper()}")
    print(f"  Baseline:  {report['baseline_mb']:.2f} MB")
    print(f"  Peak:      {report['peak_mb']:.2f} MB")
    print(f"  Limit:     {result.limit_mb:.0f} MB")
    print(f"  Headroom:  {result.headroom_mb:.1f} MB")
    print("-" * 60)

    if report["layers"]:
        print("  Per-Layer Memory Deltas:")
        for layer in report["layers"]:
            sign = "+" if layer["delta_mb"] >= 0 else ""
            print(f"    {layer['name']:.<30s} {sign}{layer['delta_mb']:.2f} MB")
        print("-" * 60)

    verdict = "✓ PASS" if result.passed else "✗ FAIL"
    print(f"  Verdict:   {verdict}")
    print(f"  {result.summary}")
    print("=" * 60 + "\n")


# ==============================================================================
# --- Main ---
# ==============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Faurge VRAM/RAM Budget Audit"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to the model to profile (e.g. models/fabian.h5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use a synthetic tensor workload (no real model needed)",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.model_path:
        parser.error("Provide --model-path or use --dry-run")

    # ---- Setup ----
    policy = load_budget_policy()
    tracker = MemoryTracker()
    tracker.snapshot("baseline")

    # ---- Profile ----
    start = time.monotonic()
    try:
        if args.dry_run:
            _run_synthetic_workload(tracker, policy)
        else:
            _run_model_workload(args.model_path, tracker, policy)
    except BudgetExceededError:
        raise  # re-raise after printing report below
    except Exception as e:
        log.error("Profiling failed: %s", e)
        return 1

    elapsed = time.monotonic() - start
    log.info("Profiling completed in %.2f seconds.", elapsed)

    # ---- Enforce ----
    report = tracker.report()
    try:
        result = enforce_budget(report["peak_mb"], policy)
    except BudgetExceededError as exc:
        _print_report(report, exc.result)
        tracker.shutdown()
        return 1

    _print_report(report, result)
    tracker.shutdown()

    # ---- JSON artifact for CI ----
    artifact_path = PROJECT_ROOT / "state" / "budget_audit.json"
    artifact = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_sec": round(elapsed, 2),
        "report": report,
        "result": {
            "passed": result.passed,
            "peak_mb": result.peak_mb,
            "limit_mb": result.limit_mb,
            "headroom_mb": result.headroom_mb,
            "overrun_mb": result.overrun_mb,
            "mode": result.mode,
            "summary": result.summary,
        },
    }
    artifact_path.write_text(json.dumps(artifact, indent=2))
    log.info("Audit artifact written to %s", artifact_path)

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
