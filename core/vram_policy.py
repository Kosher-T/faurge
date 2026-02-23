"""
core/vram_policy.py
VRAM / RAM budget policy engine for Faurge Phase 2.

Defines configurable memory ceilings and provides a hard gate that
prevents Phase 3 from proceeding if peak memory exceeds the safety limit.
In CPU-only mode the VRAM audit is skipped and RAM budget limits are
enforced instead.
"""

from dataclasses import dataclass
from typing import Optional

from core import defaults
from core.hardware_detect import HARDWARE
from core.logging import get_logger

log = get_logger("faurge.vram_policy")


# ==============================================================================
# --- Exceptions ---
# ==============================================================================

class BudgetExceededError(RuntimeError):
    """Raised when measured peak memory exceeds the configured budget."""

    def __init__(self, result: "BudgetResult"):
        self.result = result
        super().__init__(
            f"Budget EXCEEDED: peak {result.peak_mb:.1f} MB > "
            f"limit {result.limit_mb:.1f} MB "
            f"(overrun: {result.overrun_mb:.1f} MB)"
        )


# ==============================================================================
# --- Data Classes ---
# ==============================================================================

@dataclass(frozen=True)
class BudgetPolicy:
    """
    Immutable snapshot of the active memory budget rules.

    Attributes:
        mode:           'gpu' or 'cpu_ram'
        limit_mb:       Hard ceiling in MB (peak must not exceed this)
        safety_margin_mb: Extra headroom subtracted from total capacity
        abort_on_exceed: If True, enforce_budget() raises BudgetExceededError
        inference_steps: Profiling inference iterations
        warmup_steps:    Discarded warmup iterations
    """
    mode: str
    limit_mb: int
    safety_margin_mb: int
    abort_on_exceed: bool
    inference_steps: int
    warmup_steps: int


@dataclass(frozen=True)
class BudgetResult:
    """
    Outcome of a budget check.

    Attributes:
        passed:      True if peak memory stayed within the limit
        peak_mb:     Measured peak memory usage in MB
        limit_mb:    The ceiling that was applied
        headroom_mb: How much room is left (negative = overrun)
        overrun_mb:  Positive if over budget, else 0.0
        mode:        'gpu' or 'cpu_ram'
        summary:     Human-readable one-line verdict
    """
    passed: bool
    peak_mb: float
    limit_mb: float
    headroom_mb: float
    overrun_mb: float
    mode: str
    summary: str


# ==============================================================================
# --- Policy Loader ---
# ==============================================================================

def load_budget_policy(
    *,
    vram_limit: Optional[int] = None,
    ram_limit: Optional[int] = None,
    safety_margin: Optional[int] = None,
    abort: Optional[bool] = None,
    inference_steps: Optional[int] = None,
    warmup_steps: Optional[int] = None,
) -> BudgetPolicy:
    """
    Build a BudgetPolicy from the current hardware profile and settings.

    All parameters default to the values in core.settings (which in turn
    fall back to core.defaults).  Explicit keyword arguments override them,
    useful for testing or one-off CLI overrides.
    """
    # Lazy import to avoid circular dependency at module level
    from core import settings

    if HARDWARE.cpu_only_mode:
        mode = "cpu_ram"
        limit = ram_limit if ram_limit is not None else settings.RAM_BUDGET_LIMIT_MB
    else:
        mode = "gpu"
        limit = vram_limit if vram_limit is not None else settings.VRAM_BUDGET_LIMIT_MB

    margin = safety_margin if safety_margin is not None else settings.VRAM_SAFETY_MARGIN_MB
    do_abort = abort if abort is not None else settings.BUDGET_ABORT_ON_EXCEED
    inf_steps = inference_steps if inference_steps is not None else settings.BUDGET_INFERENCE_STEPS
    warm_steps = warmup_steps if warmup_steps is not None else settings.BUDGET_WARMUP_STEPS

    policy = BudgetPolicy(
        mode=mode,
        limit_mb=limit,
        safety_margin_mb=margin,
        abort_on_exceed=do_abort,
        inference_steps=inf_steps,
        warmup_steps=warm_steps,
    )

    log.info(
        "Budget policy loaded — mode=%s, limit=%d MB, abort=%s",
        policy.mode, policy.limit_mb, policy.abort_on_exceed,
    )
    return policy


# ==============================================================================
# --- Budget Check ---
# ==============================================================================

def check_budget(peak_usage_mb: float, policy: BudgetPolicy) -> BudgetResult:
    """
    Compare measured peak usage against the policy ceiling.

    Returns a BudgetResult without raising; callers decide how to act.
    """
    headroom = policy.limit_mb - peak_usage_mb
    overrun = max(0.0, -headroom)
    passed = peak_usage_mb <= policy.limit_mb

    if passed:
        summary = (
            f"PASS — peak {peak_usage_mb:.1f} MB within "
            f"{policy.limit_mb} MB limit ({headroom:.1f} MB headroom)"
        )
    else:
        summary = (
            f"FAIL — peak {peak_usage_mb:.1f} MB exceeds "
            f"{policy.limit_mb} MB limit by {overrun:.1f} MB"
        )

    return BudgetResult(
        passed=passed,
        peak_mb=peak_usage_mb,
        limit_mb=float(policy.limit_mb),
        headroom_mb=headroom,
        overrun_mb=overrun,
        mode=policy.mode,
        summary=summary,
    )


# ==============================================================================
# --- Enforcement ---
# ==============================================================================

def enforce_budget(
    peak_usage_mb: float,
    policy: Optional[BudgetPolicy] = None,
) -> BudgetResult:
    """
    Convenience function: load policy (if not provided), check the budget,
    and raise BudgetExceededError when abort_on_exceed is True.

    Returns the BudgetResult on success or if abort is disabled.
    """
    if policy is None:
        policy = load_budget_policy()

    result = check_budget(peak_usage_mb, policy)

    if result.passed:
        log.info("✓ Budget check passed: %s", result.summary)
    else:
        log.warning("✗ Budget check failed: %s", result.summary)
        if policy.abort_on_exceed:
            raise BudgetExceededError(result)

    return result
