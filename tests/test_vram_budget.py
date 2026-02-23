"""
tests/test_vram_budget.py
Comprehensive pytest suite for Phase 2 Memory & Budget Enforcement.

Validates:
  - BudgetPolicy loading (GPU and CPU-only modes)
  - Budget check logic (pass / fail)
  - Budget enforcement (abort vs. warn-only)
  - MemoryTracker snapshots, layer tracking, and peak calculation

All GPU-dependent paths are mocked so tests run in CI without a GPU.
"""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def _mock_hardware_gpu():
    """Patch HARDWARE to simulate a GPU-enabled system."""
    mock_hw = MagicMock()
    mock_hw.cpu_only_mode = False
    mock_hw.gpu_available = True
    mock_hw.gpu.name = "Mock GPU"
    mock_hw.gpu.vram_total_mb = 4096
    with patch("core.vram_policy.HARDWARE", mock_hw), \
         patch("core.memory_tracker.HARDWARE", mock_hw):
        yield mock_hw


@pytest.fixture
def _mock_hardware_cpu():
    """Patch HARDWARE to simulate a CPU-only system."""
    mock_hw = MagicMock()
    mock_hw.cpu_only_mode = True
    mock_hw.gpu_available = False
    mock_hw.gpu = None
    with patch("core.vram_policy.HARDWARE", mock_hw), \
         patch("core.memory_tracker.HARDWARE", mock_hw):
        yield mock_hw


# ===========================================================================
# --- BudgetPolicy Tests ---
# ===========================================================================

class TestBudgetPolicy:
    """Tests for load_budget_policy()."""

    def test_budget_policy_gpu_mode(self, _mock_hardware_gpu):
        """In GPU mode, policy uses VRAM limit and reports mode='gpu'."""
        from core.vram_policy import load_budget_policy

        policy = load_budget_policy(vram_limit=3900)
        assert policy.mode == "gpu"
        assert policy.limit_mb == 3900
        assert policy.abort_on_exceed is True

    def test_budget_policy_cpu_mode(self, _mock_hardware_cpu):
        """In CPU-only mode, policy uses RAM limit and reports mode='cpu_ram'."""
        from core.vram_policy import load_budget_policy

        policy = load_budget_policy(ram_limit=4096)
        assert policy.mode == "cpu_ram"
        assert policy.limit_mb == 4096

    def test_budget_policy_custom_overrides(self, _mock_hardware_gpu):
        """Explicit keyword arguments override defaults."""
        from core.vram_policy import load_budget_policy

        policy = load_budget_policy(
            vram_limit=2048,
            safety_margin=200,
            abort=False,
            inference_steps=10,
            warmup_steps=3,
        )
        assert policy.limit_mb == 2048
        assert policy.safety_margin_mb == 200
        assert policy.abort_on_exceed is False
        assert policy.inference_steps == 10
        assert policy.warmup_steps == 3


# ===========================================================================
# --- Budget Check Tests ---
# ===========================================================================

class TestCheckBudget:
    """Tests for check_budget()."""

    def test_check_budget_passes(self, _mock_hardware_gpu):
        """Usage under limit → passed=True, positive headroom."""
        from core.vram_policy import load_budget_policy, check_budget

        policy = load_budget_policy(vram_limit=4000)
        result = check_budget(3500.0, policy)

        assert result.passed is True
        assert result.peak_mb == 3500.0
        assert result.limit_mb == 4000.0
        assert result.headroom_mb == 500.0
        assert result.overrun_mb == 0.0
        assert "PASS" in result.summary

    def test_check_budget_fails(self, _mock_hardware_gpu):
        """Usage over limit → passed=False, positive overrun."""
        from core.vram_policy import load_budget_policy, check_budget

        policy = load_budget_policy(vram_limit=3900)
        result = check_budget(4100.0, policy)

        assert result.passed is False
        assert result.peak_mb == 4100.0
        assert result.overrun_mb == 200.0
        assert result.headroom_mb == -200.0
        assert "FAIL" in result.summary

    def test_check_budget_exact_limit(self, _mock_hardware_gpu):
        """Usage exactly at limit → passes (<=)."""
        from core.vram_policy import load_budget_policy, check_budget

        policy = load_budget_policy(vram_limit=3900)
        result = check_budget(3900.0, policy)

        assert result.passed is True
        assert result.headroom_mb == 0.0


# ===========================================================================
# --- Budget Enforcement Tests ---
# ===========================================================================

class TestEnforceBudget:
    """Tests for enforce_budget()."""

    def test_enforce_budget_raises(self, _mock_hardware_gpu):
        """Exceeding budget with abort=True raises BudgetExceededError."""
        from core.vram_policy import (
            BudgetExceededError,
            enforce_budget,
            load_budget_policy,
        )

        policy = load_budget_policy(vram_limit=3900, abort=True)

        with pytest.raises(BudgetExceededError) as exc_info:
            enforce_budget(4200.0, policy)

        assert exc_info.value.result.passed is False
        assert exc_info.value.result.overrun_mb == 300.0

    def test_enforce_budget_no_abort(self, _mock_hardware_gpu):
        """Exceeding budget with abort=False logs warning but returns result."""
        from core.vram_policy import enforce_budget, load_budget_policy

        policy = load_budget_policy(vram_limit=3900, abort=False)
        result = enforce_budget(4200.0, policy)

        assert result.passed is False
        assert result.overrun_mb == 300.0
        # Should not raise — returning normally means success

    def test_enforce_budget_passes(self, _mock_hardware_gpu):
        """Under budget → returns passing result, no exception."""
        from core.vram_policy import enforce_budget, load_budget_policy

        policy = load_budget_policy(vram_limit=4000, abort=True)
        result = enforce_budget(3000.0, policy)

        assert result.passed is True
        assert result.headroom_mb == 1000.0


# ===========================================================================
# --- MemoryTracker Tests ---
# ===========================================================================

class TestMemoryTracker:
    """Tests for MemoryTracker using mocked psutil backend."""

    def _make_tracker(self):
        """Create a CPU-mode tracker with mocked psutil."""
        from core.memory_tracker import MemoryTracker
        return MemoryTracker(mode="cpu_ram")

    @patch("core.memory_tracker.MemoryTracker._read_ram_mb", return_value=1024.0)
    def test_snapshot_returns_valid_data(self, mock_ram, _mock_hardware_cpu):
        """snapshot() returns a MemorySnapshot with sensible values."""
        tracker = self._make_tracker()
        snap = tracker.snapshot("test_label")

        assert snap.used_mb == 1024.0
        assert snap.mode == "cpu_ram"
        assert snap.label == "test_label"
        assert snap.timestamp > 0

    @patch("core.memory_tracker.MemoryTracker._read_ram_mb")
    def test_layer_tracking(self, mock_ram, _mock_hardware_cpu):
        """track_layer() records named memory deltas."""
        mock_ram.side_effect = [1000.0, 1256.0]  # before, after

        tracker = self._make_tracker()
        with tracker.track_layer("my_layer"):
            pass  # simulated work

        report = tracker.report()
        assert len(report["layers"]) == 1
        assert report["layers"][0]["name"] == "my_layer"
        assert report["layers"][0]["delta_mb"] == 256.0

    @patch("core.memory_tracker.MemoryTracker._read_ram_mb")
    def test_peak_tracking(self, mock_ram, _mock_hardware_cpu):
        """peak_mb returns the highest observed reading."""
        # Three snapshots: 1000, 2500, 1800
        mock_ram.side_effect = [1000.0, 2500.0, 1800.0]

        tracker = self._make_tracker()
        tracker.snapshot("low")
        tracker.snapshot("high")
        tracker.snapshot("mid")

        assert tracker.peak_mb == 2500.0

    @patch("core.memory_tracker.MemoryTracker._read_ram_mb")
    def test_report_structure(self, mock_ram, _mock_hardware_cpu):
        """report() returns a dict with the expected keys."""
        mock_ram.return_value = 512.0

        tracker = self._make_tracker()
        tracker.snapshot("baseline")

        report = tracker.report()
        assert "mode" in report
        assert "peak_mb" in report
        assert "baseline_mb" in report
        assert "snapshots" in report
        assert "layers" in report
        assert report["snapshot_count"] == 1
        assert report["baseline_mb"] == 512.0
