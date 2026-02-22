#!/usr/bin/env python3
"""
scripts/pipewire_compat_check.py
Checks installed PipeWire, WirePlumber, and JACK versions against a
known-stable compatibility matrix for Faurge.
Exits non-zero if any component is below the minimum supported version.
"""

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Compatibility Matrix
# ---------------------------------------------------------------------------

@dataclass
class CompatEntry:
    """One row in the compatibility matrix."""
    component: str
    binary: str
    version_cmd: list[str]      # full command (may differ from binary)
    min_version: tuple[int, ...]
    min_version_str: str
    notes: str = ""


COMPAT_MATRIX: list[CompatEntry] = [
    CompatEntry(
        component="PipeWire",
        binary="pw-cli",
        version_cmd=["pw-cli", "--version"],
        min_version=(0, 3, 60),
        min_version_str="0.3.60",
        notes="Core audio server; required for loopback and routing.",
    ),
    CompatEntry(
        component="WirePlumber",
        binary="wpctl",
        # wpctl has no --version flag; `wpctl status` prints the version
        # in the first line, e.g. "PipeWire 'pipewire-0' [1.5.85, ...]"
        # and WirePlumber's own version under Clients.
        version_cmd=["wpctl", "status"],
        min_version=(0, 4, 14),
        min_version_str="0.4.14",
        notes="Session/policy manager for PipeWire.",
    ),
    CompatEntry(
        component="PipeWire JACK",
        binary="pw-jack",
        # pw-jack has no version flag; use `pipewire --version` which
        # reports the core libpipewire version (JACK layer ships together).
        version_cmd=["pipewire", "--version"],
        min_version=(0, 3, 60),
        min_version_str="0.3.60",
        notes="JACK compatibility layer for LV2 plugin hosting.",
    ),
]

# ANSI colours
_GREEN = "\033[0;32m"
_RED = "\033[0;31m"
_YELLOW = "\033[1;33m"
_CYAN = "\033[0;36m"
_BOLD = "\033[1m"
_NC = "\033[0m"


# ---------------------------------------------------------------------------
# Version helpers
# ---------------------------------------------------------------------------

_VERSION_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def _parse_version(raw: str) -> Optional[tuple[int, ...]]:
    """Extract the first semver-like triplet from a string."""
    m = _VERSION_RE.search(raw)
    if m:
        return tuple(int(g) for g in m.groups())
    return None


def _get_version_output(cmd: list[str]) -> Optional[str]:
    """Run `cmd` and return stripped stdout+stderr, or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5,
        )
        # Some tools print version to stderr instead of stdout
        output = result.stdout.strip() or result.stderr.strip()
        return output if output else None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


# ---------------------------------------------------------------------------
# Check runner
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    component: str
    installed: bool
    version_raw: Optional[str]
    version_parsed: Optional[tuple[int, ...]]
    min_version_str: str
    passed: bool
    notes: str


def check_component(entry: CompatEntry) -> CheckResult:
    """Check a single component against the matrix."""
    # First check if the primary binary exists on PATH
    if shutil.which(entry.binary) is None:
        return CheckResult(
            component=entry.component,
            installed=False,
            version_raw=None,
            version_parsed=None,
            min_version_str=entry.min_version_str,
            passed=False,
            notes=f"{entry.binary} not found on PATH",
        )

    raw = _get_version_output(entry.version_cmd)

    if raw is None:
        return CheckResult(
            component=entry.component,
            installed=False,
            version_raw=None,
            version_parsed=None,
            min_version_str=entry.min_version_str,
            passed=False,
            notes=f"{entry.binary} not found on PATH",
        )

    parsed = _parse_version(raw)
    if parsed is None:
        return CheckResult(
            component=entry.component,
            installed=True,
            version_raw=raw,
            version_parsed=None,
            min_version_str=entry.min_version_str,
            passed=False,
            notes=f"Could not parse version from: {raw!r}",
        )

    ok = parsed >= entry.min_version
    return CheckResult(
        component=entry.component,
        installed=True,
        version_raw=raw,
        version_parsed=parsed,
        min_version_str=entry.min_version_str,
        passed=ok,
        notes=entry.notes if ok else f"Installed {'.'.join(map(str, parsed))} < required {entry.min_version_str}",
    )


def run_all() -> list[CheckResult]:
    return [check_component(entry) for entry in COMPAT_MATRIX]


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_table(results: list[CheckResult]) -> bool:
    """Pretty-print a compatibility table. Returns True if all passed."""
    hdr = f"{'Component':<20s} {'Installed':<12s} {'Detected':<14s} {'Minimum':<12s} {'Status':<10s}"

    print(f"\n{_BOLD}{'='*70}")
    print(f" Faurge — PipeWire Compatibility Check")
    print(f"{'='*70}{_NC}\n")
    print(f"  {_CYAN}{hdr}{_NC}")
    print(f"  {'─'*68}")

    all_ok = True
    for r in results:
        detected = ".".join(map(str, r.version_parsed)) if r.version_parsed else "—"
        installed = f"{_GREEN}yes{_NC}" if r.installed else f"{_RED}no{_NC}"

        if r.passed:
            status = f"{_GREEN}✓ PASS{_NC}"
        else:
            status = f"{_RED}✗ FAIL{_NC}"
            all_ok = False

        # Plain widths (without ANSI) for alignment
        print(f"  {r.component:<20s} {installed}{'':>{6 if r.installed else 7}s} {detected:<14s} {r.min_version_str:<12s} {status}")

    print(f"\n  {'─'*68}")
    for r in results:
        if not r.passed:
            print(f"  {_RED}▸{_NC} {r.component}: {r.notes}")

    print()
    if all_ok:
        print(f"  {_GREEN}All components meet the compatibility requirements.{_NC}")
    else:
        print(f"  {_RED}One or more components failed. Please upgrade before running Faurge.{_NC}")
    print(f"\n{_BOLD}{'='*70}{_NC}\n")

    return all_ok


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    results = run_all()
    all_ok = print_table(results)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
