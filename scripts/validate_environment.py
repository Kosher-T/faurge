#!/usr/bin/env python3
"""
scripts/validate_environment.py
Hard-fail preflight checks for the Faurge runtime environment.
Exits non-zero on any critical failure; outputs a structured JSON report.
"""

import importlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MIN_PYTHON = (3, 11)
MIN_DISK_FREE_GB = 2.0

REQUIRED_PYTHON_PACKAGES = [
    "numpy",
    "scipy",
    "soundfile",
    "librosa",
    "pynvml",
    "dotenv",
    "jsonschema",
]

REQUIRED_SYSTEM_BINARIES = [
    "pw-cli",
    "pw-link",
    "pw-loopback",
    "wpctl",
    "pactl",
]

# ANSI colours
_GREEN = "\033[0;32m"
_RED = "\033[0;31m"
_YELLOW = "\033[1;33m"
_NC = "\033[0m"


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def check_python_version() -> dict:
    """Verify Python >= MIN_PYTHON."""
    current = sys.version_info[:2]
    ok = current >= MIN_PYTHON
    return {
        "name": "python_version",
        "passed": ok,
        "detail": f"{current[0]}.{current[1]} (need >={MIN_PYTHON[0]}.{MIN_PYTHON[1]})",
    }


def check_python_packages() -> dict:
    """Attempt to import every required Python package."""
    missing: list[str] = []
    for pkg in REQUIRED_PYTHON_PACKAGES:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)

    return {
        "name": "python_packages",
        "passed": len(missing) == 0,
        "detail": "all present" if not missing else f"missing: {', '.join(missing)}",
    }


def check_system_binaries() -> dict:
    """Verify required system-level binaries are on PATH."""
    missing: list[str] = []
    for binary in REQUIRED_SYSTEM_BINARIES:
        if shutil.which(binary) is None:
            missing.append(binary)

    return {
        "name": "system_binaries",
        "passed": len(missing) == 0,
        "detail": "all present" if not missing else f"missing: {', '.join(missing)}",
    }


def check_gpu_driver() -> dict:
    """
    Import hardware_detect and inspect the profile.
    This is a WARNING-level check — CPU-only mode is valid.
    """
    try:
        # Ensure project root is on sys.path so `core.*` resolves
        project_root = str(Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from core.hardware_detect import HARDWARE

        if HARDWARE.gpu_available:
            detail = f"GPU detected: {HARDWARE.gpu.name} ({HARDWARE.gpu.vram_total_mb} MB)"
        else:
            detail = "No GPU — CPU-only mode (valid, but VRAM watchdog disabled)"

        return {
            "name": "gpu_driver",
            "passed": True,  # never a hard failure
            "warning": not HARDWARE.gpu_available,
            "detail": detail,
        }
    except Exception as exc:
        return {
            "name": "gpu_driver",
            "passed": True,
            "warning": True,
            "detail": f"Could not probe hardware: {exc}",
        }


def check_disk_space() -> dict:
    """Ensure at least MIN_DISK_FREE_GB of free space on the project partition."""
    project_root = Path(__file__).resolve().parent.parent
    usage = shutil.disk_usage(project_root)
    free_gb = round(usage.free / (1024 ** 3), 2)
    ok = free_gb >= MIN_DISK_FREE_GB
    return {
        "name": "disk_space",
        "passed": ok,
        "detail": f"{free_gb} GB free (need >={MIN_DISK_FREE_GB} GB)",
    }


def check_settings_schema() -> dict:
    """Run the JSON-Schema validation on the current Faurge settings."""
    try:
        project_root = str(Path(__file__).resolve().parent.parent)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        from core.settings import validate_settings
        validate_settings()
        return {
            "name": "settings_schema",
            "passed": True,
            "detail": "settings_schema_v1.json validated OK",
        }
    except RuntimeError as exc:
        return {
            "name": "settings_schema",
            "passed": False,
            "detail": str(exc),
        }
    except Exception as exc:
        return {
            "name": "settings_schema",
            "passed": False,
            "detail": f"Unexpected error: {exc}",
        }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_checks() -> list[dict]:
    """Execute every check and return a list of result dicts."""
    return [
        check_python_version(),
        check_python_packages(),
        check_system_binaries(),
        check_gpu_driver(),
        check_disk_space(),
        check_settings_schema(),
    ]


def print_report(results: list[dict]) -> bool:
    """
    Pretty-print results and return True if all critical checks passed.
    """
    print(f"\n{'='*60}")
    print(f" Faurge Environment Validation")
    print(f"{'='*60}\n")

    all_passed = True
    for r in results:
        is_warning = r.get("warning", False)
        if r["passed"] and not is_warning:
            icon = f"{_GREEN}✓ PASS{_NC}"
        elif r["passed"] and is_warning:
            icon = f"{_YELLOW}⚠ WARN{_NC}"
        else:
            icon = f"{_RED}✗ FAIL{_NC}"
            all_passed = False

        print(f"  {icon}  {r['name']:<20s} — {r['detail']}")

    print(f"\n{'='*60}")
    if all_passed:
        print(f"  {_GREEN}All critical checks passed.{_NC}")
    else:
        print(f"  {_RED}One or more critical checks FAILED. Fix before proceeding.{_NC}")
    print(f"{'='*60}\n")

    return all_passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    results = run_all_checks()
    all_ok = print_report(results)

    # Also dump machine-readable JSON to stdout
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "all_passed": all_ok,
        "checks": results,
    }
    print(json.dumps(report, indent=2))

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
