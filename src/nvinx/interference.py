"""Interference modeling for v0.2/v0.3 Pattern B (queue-aware substrate-native).

Per the v0.2-v0.4 calibration journey on small-VRAM mobile GPU substrates:

  - iGniter cache-pressure formula (Xu/Liu IEEE TPDS 2023) does NOT transfer to
    heterogeneous-kernel-rate substrates without real Nsight Compute measurement
  - Tetris cluster-variance metric (Xu/Liu IEEE TSC 2024) does NOT transfer to
    single-bench substrates (no spatial dimension)
  - Substrate-native queue-aware model achieves ~16% LOPO mean error on 4-model
    corpus; persistent ~30% outlier on 2-small-kernel pairs

This module provides:
  - HardwareCoefficients + InterferenceProfile dataclasses
  - predict_pair_latency_queue_aware: substrate-native queue-aware prediction
  - max_kernel_rate_score: cheap pre-filter heuristic (ρ=0.50 with slowdown)
  - asymmetry_predictor: act_solo_ratio (ρ=0.72 for which-suffers-more)
  - PairLookup: per-pair measured-ground-truth safety net for known high-error pairs

All artifacts are operator-generated only (security discipline: published reference
profiles would create a multi-tenant attack surface). Calibration tooling lives in a
separate research workspace (private); a turnkey ``nvinx.calibration`` module is a
v0.3+ goal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional


# ============================================================================
# Dataclasses (frozen; operator-generated only)
# ============================================================================


@dataclass(frozen=True)
class HardwareCoefficients:
    """Substrate-level fitted coefficients (one-time per substrate).

    Fitted from hardware sweeps (idlef polynomial + powerp linear). Used by
    the queue-aware prediction formula. Operator-generated; not caller-supplied.
    """
    idlef_polynomial: tuple[float, ...]
    powerp_linear: tuple[float, ...]
    nominal_freq_mhz: float
    tdp_watts: float
    substrate_name: str = "unknown"

    def scheduling_delay_ms_at_concurrency(self, n_concurrent: int) -> float:
        """Per-inference scheduling delay added at N concurrent workloads.

        Per v0.4 finding: per-inference offset, NOT per-kernel × kernel_count.
        Returns 0.0 for N=1 (no concurrency).
        """
        if n_concurrent < 2:
            return 0.0
        # Polynomial evaluated at N (highest order first)
        result = 0.0
        for coef in self.idlef_polynomial:
            result = result * n_concurrent + coef
        return max(result, 0.0)


@dataclass(frozen=True)
class InterferenceProfile:
    """Per-model interference coefficients (operator-profiled).

    Fitted from real Nsight Compute L2 saturation measurement + standalone
    inference profiling. The `theta` queue-aware coefficient is fitted via
    least-squares on cross-pair calibration measurements.

    Per Option B-extended (delve §2.1): operator-generated only; not user-
    supplied. Queue-aware predictions silently fall back to None-theta if
    profile lacks theta (calibration not yet performed).
    """
    name: str
    kernels: int
    baseidle_ms: float
    act_solo_ms: float
    l2_saturation_pct: float  # via real ncu (lts__t_sectors.avg.pct_of_peak_sustained_elapsed)
    k_l2: float = 0.0  # legacy iGniter cache-sensitivity (kept for reference; not load-bearing)
    theta: Optional[float] = None  # queue-aware (fitted from cross-pair calibration)
    power_w: float = 0.0
    architecture_class: str = "unknown"  # e.g. "encoder_transformer", "decoder_transformer", "encoder_decoder"

    @property
    def kernel_rate(self) -> float:
        """Kernels per ms (high → small fast kernels; low → big slow kernels)."""
        if self.act_solo_ms == 0:
            return 0.0
        return self.kernels / self.act_solo_ms

    @property
    def kernel_duration_ms(self) -> float:
        """Average kernel duration in ms."""
        if self.kernels == 0:
            return 0.0
        return self.act_solo_ms / self.kernels


@dataclass(frozen=True)
class PairLookupEntry:
    """Measured ground-truth latency for a known co-located pair.

    Used as safety net for pairs where queue-aware predictor has known high
    error (e.g., 2-small-kernel pairs like short+qwen with ~30% LOPO error).
    Per delve §5.1: preserves measured ground truth at cost of generalization.
    """
    pair: tuple[str, str]
    measured_latency_a_ms: float
    measured_latency_b_ms: float
    measured_at: str = ""  # ISO 8601 UTC


# ============================================================================
# Queue-aware prediction (v0.3 D2 + v0.4 final)
# ============================================================================


def predict_pair_latency_queue_aware(
    profile_a: InterferenceProfile,
    profile_b: InterferenceProfile,
    hw: HardwareCoefficients,
) -> tuple[float, float]:
    """Substrate-native queue-aware interference prediction.

    Formula (per v0.3 D2 derivation):
        latency_i = act_solo_i × (1 + theta_i × partner_act / (act_solo_i + partner_act))

    Plus per-inference scheduling delay (NOT per-kernel scaled, per v0.4 fix).

    Both profiles must have non-None theta (fitted from cross-pair calibration).

    Returns (latency_a_ms, latency_b_ms). LOPO mean error ~16% on 4-model
    corpus (RTX A1000 mobile substrate; transformer/LLM workload mix);
    expect ~30% on 2-small-kernel-rate pairs (formula limit).
    """
    if profile_a.theta is None or profile_b.theta is None:
        raise ValueError(
            f"predict_pair_latency_queue_aware requires fitted theta values; "
            f"got theta_a={profile_a.theta}, theta_b={profile_b.theta}"
        )

    sum_act = profile_a.act_solo_ms + profile_b.act_solo_ms
    if sum_act == 0:
        return 0.0, 0.0

    # Asymmetric partner-time-fraction inflation
    fraction_a = profile_b.act_solo_ms / sum_act
    fraction_b = profile_a.act_solo_ms / sum_act
    base_a = profile_a.act_solo_ms * (1.0 + profile_a.theta * fraction_a)
    base_b = profile_b.act_solo_ms * (1.0 + profile_b.theta * fraction_b)

    # Per-inference scheduling delay (constant per inference under concurrency)
    sched_delay = hw.scheduling_delay_ms_at_concurrency(2)
    pred_a = profile_a.baseidle_ms + sched_delay + base_a
    pred_b = profile_b.baseidle_ms + sched_delay + base_b
    return pred_a, pred_b


# ============================================================================
# Heuristic augmentations (v0.4 alpha findings)
# ============================================================================


def max_kernel_rate_score(profiles: list[InterferenceProfile]) -> float:
    """Pre-filter heuristic: max kernel rate across placement candidates.

    Empirical Spearman correlation ρ=+0.50 with measured max_slowdown across
    10-pair corpus (v0.4 alpha). High max_kernel_rate → expect significant
    queue contention. Cheap first-pass filter; not as accurate as queue-aware
    formula but doesn't require fitted theta.
    """
    if not profiles:
        return 0.0
    return max(p.kernel_rate for p in profiles)


def asymmetry_predictor(
    profile_a: InterferenceProfile,
    profile_b: InterferenceProfile,
) -> float:
    """Predict which model in pair suffers more (act_solo_ratio).

    Returns max(act_solo) / min(act_solo). Empirical Spearman correlation
    ρ=+0.72 with measured asymmetry across 10-pair corpus (v0.4 alpha).
    Higher ratio → more asymmetric slowdown (smaller model suffers more).
    """
    a = profile_a.act_solo_ms
    b = profile_b.act_solo_ms
    if min(a, b) == 0:
        return float("inf")
    return max(a, b) / min(a, b)


# ============================================================================
# Per-pair lookup safety net (delve §5.1)
# ============================================================================


def lookup_pair_latency(
    pair: tuple[str, str],
    lookup: Mapping[tuple[str, str], PairLookupEntry],
) -> Optional[tuple[float, float]]:
    """Try lookup for a known co-located pair.

    Returns measured latencies (a_ms, b_ms) if pair is in lookup, else None.
    Pair tuple is canonicalized (sorted) before lookup.
    """
    a_name, b_name = pair
    canonical = tuple(sorted([a_name, b_name]))
    entry = lookup.get(canonical)
    if entry is None:
        return None
    # Re-orient to match input order
    if (a_name, b_name) == entry.pair:
        return entry.measured_latency_a_ms, entry.measured_latency_b_ms
    elif (b_name, a_name) == entry.pair:
        return entry.measured_latency_b_ms, entry.measured_latency_a_ms
    # Fallback: return in input order assuming lookup canonical = input canonical
    if a_name == canonical[0]:
        return entry.measured_latency_a_ms, entry.measured_latency_b_ms
    return entry.measured_latency_b_ms, entry.measured_latency_a_ms


def predict_pair_latency(
    profile_a: InterferenceProfile,
    profile_b: InterferenceProfile,
    hw: HardwareCoefficients,
    *,
    pair_lookup: Optional[Mapping[tuple[str, str], PairLookupEntry]] = None,
) -> tuple[float, float, str]:
    """Tiered pair latency prediction: lookup first, then queue-aware.

    Returns (latency_a_ms, latency_b_ms, source) where source is one of:
      - "lookup": measured ground truth from pair_lookup
      - "queue_aware": fitted queue-aware formula prediction
      - "fallback_unknown": both models lack theta; returned as standalone sum
    """
    pair = (profile_a.name, profile_b.name)
    if pair_lookup is not None:
        result = lookup_pair_latency(pair, pair_lookup)
        if result is not None:
            return result[0], result[1], "lookup"

    if profile_a.theta is not None and profile_b.theta is not None:
        a, b = predict_pair_latency_queue_aware(profile_a, profile_b, hw)
        return a, b, "queue_aware"

    # No theta + no lookup: cannot predict; return standalone (no interference)
    return profile_a.act_solo_ms, profile_b.act_solo_ms, "fallback_unknown"
