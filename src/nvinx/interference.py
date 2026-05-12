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

from collections.abc import Mapping
from dataclasses import dataclass

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
    theta: float | None = None  # queue-aware (fitted from cross-pair calibration)
    power_w: float = 0.0
    architecture_class: str = (
        "unknown"  # e.g. "encoder_transformer", "decoder_transformer", "encoder_decoder"
    )

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
# V5 kernel-size-ratio correction (v0.6 alpha; substrate-bound)
# ============================================================================


def predict_pair_latency_queue_aware_v5(
    profile_a: InterferenceProfile,
    profile_b: InterferenceProfile,
    hw: HardwareCoefficients,
    *,
    gamma_kernel_size: float | None = None,
) -> tuple[float, float]:
    """Substrate-native queue-aware with kernel-size-ratio correction.

    Extends :func:`predict_pair_latency_queue_aware` with one optional empirical
    parameter `gamma_kernel_size` capturing the asymmetry between small-kernel
    self models and large-kernel partner models. Formula::

        partner_frac      = act_partner / (act_self + act_partner)
        kernel_size_ratio = (act_partner / kernels_partner) / (act_self / kernels_self)
        latency_self      = act_self × (1 + theta_self × partner_frac
                                          × (1 + gamma × kernel_size_ratio))
                          + scheduling_delay(N_concurrent=2)
                          + baseidle_self

    When ``gamma_kernel_size`` is ``None`` or ``0.0``, this function is
    numerically identical to :func:`predict_pair_latency_queue_aware`. When
    supplied (operator-fitted on the operator's substrate), the kernel-size-
    ratio correction is applied.

    SUBSTRATE-BOUND.  ``gamma_kernel_size`` is empirical per-substrate. The
    reference value γ ≈ 0.75 was fitted on the v0.5 7-model 19-pair corpus
    on a 4 GB RTX A1000 mobile substrate with a transformer/LLM workload mix
    (LOPO mean error 18.4% vs queue-aware baseline 23.3% on that corpus).
    Operators on other substrates MUST refit via :func:`fit_gamma_kernel_size`.
    Using the reference value uncalibrated may DEGRADE predictions on
    substrate classes where the kernel-size-ratio effect does not dominate
    (e.g. datacenter-class GPUs with many SMs where cache contention is the
    dominant interference physics).

    Both profiles must have non-None theta (fitted from cross-pair calibration).
    """
    if profile_a.theta is None or profile_b.theta is None:
        raise ValueError(
            f"predict_pair_latency_queue_aware_v5 requires fitted theta values; "
            f"got theta_a={profile_a.theta}, theta_b={profile_b.theta}"
        )

    sum_act = profile_a.act_solo_ms + profile_b.act_solo_ms
    if sum_act == 0:
        return 0.0, 0.0

    fraction_a = profile_b.act_solo_ms / sum_act
    fraction_b = profile_a.act_solo_ms / sum_act

    if gamma_kernel_size is None or gamma_kernel_size == 0.0:
        # Backward-compat with queue-aware v0.2: identical numeric output
        v5_factor_a = 1.0
        v5_factor_b = 1.0
    else:
        kd_a = profile_a.kernel_duration_ms
        kd_b = profile_b.kernel_duration_ms
        if kd_a == 0 or kd_b == 0:
            v5_factor_a = 1.0
            v5_factor_b = 1.0
        else:
            # kernel_size_ratio_self = partner-side-kernel-duration / self-side-kernel-duration
            ksr_a = kd_b / kd_a
            ksr_b = kd_a / kd_b
            v5_factor_a = 1.0 + gamma_kernel_size * ksr_a
            v5_factor_b = 1.0 + gamma_kernel_size * ksr_b

    base_a = profile_a.act_solo_ms * (1.0 + profile_a.theta * fraction_a * v5_factor_a)
    base_b = profile_b.act_solo_ms * (1.0 + profile_b.theta * fraction_b * v5_factor_b)

    sched_delay = hw.scheduling_delay_ms_at_concurrency(2)
    pred_a = profile_a.baseidle_ms + sched_delay + base_a
    pred_b = profile_b.baseidle_ms + sched_delay + base_b
    return pred_a, pred_b


def fit_gamma_kernel_size(
    profiles: Mapping[str, InterferenceProfile],
    pair_measurements: list[tuple[str, str, float, float]],
    hw: HardwareCoefficients,
) -> float:
    """Fit substrate-specific ``gamma_kernel_size`` via 1-parameter relative-residual LS.

    Closed-form weighted 1-D regression (no numpy dependency): minimises the
    sum of squared *relative* residuals ``((pred − observed) / observed)²``,
    matching the residual convention used by the reference v0.5 H5 joint-fit
    (scipy.optimize.least_squares on relative residuals; see
    research/v0_2_calibration/track_c_h5_short_act_solo.py). The closed-form
    solution with theta held fixed is::

        gamma_hat = sum(w_i * x_i * y_i) / sum(w_i * x_i * x_i)

    where, for each (self, partner) direction of each calibration pair::

        y_i = observed_self - baseidle_self - sched_delay
              - act_self * (1 + theta_self * partner_frac)
        x_i = act_self * theta_self * partner_frac * kernel_size_ratio
        w_i = 1 / observed_self ** 2

    Both directions of each pair contribute (so N pairs → 2N measurements).

    Relative-residual weighting is the appropriate choice for substrates with
    wide ``act_solo`` spread (10×+ between smallest and largest model): it
    weights every pair equally in % error space rather than over-weighting
    large-act pairs as absolute-residual LS would.

    Parameters
    ----------
    profiles
        Mapping from model name to :class:`InterferenceProfile` (theta fitted).
    pair_measurements
        List of ``(name_a, name_b, observed_latency_a_ms, observed_latency_b_ms)``
        co-located measurements. ``name_a`` and ``name_b`` must be keys in
        ``profiles``.
    hw
        :class:`HardwareCoefficients` for the operator's substrate.

    Returns
    -------
    float
        Fitted gamma_kernel_size coefficient. Pass to
        :func:`predict_pair_latency_queue_aware_v5` via ``gamma_kernel_size=``.

    Raises
    ------
    ValueError
        If ``pair_measurements`` is empty, or any referenced profile lacks a
        fitted ``theta``, or the regressor sum collapses to zero (degenerate
        calibration data: all kernel_size_ratio values equal zero), or any
        observed latency is ≤ 0 (relative residual undefined).

    Notes
    -----
    SUBSTRATE-BOUND. The fitted coefficient is meaningful only for the same
    substrate (GPU model, driver, workload-class mix) on which the calibration
    measurements were collected. Re-fit if any of those change. This function
    holds theta fixed; for joint theta + gamma refit (the residual norm used
    by the v0.5 H5 reference), use scipy.optimize.least_squares with
    :func:`predict_pair_latency_queue_aware_v5` as the predictor.
    """
    if not pair_measurements:
        raise ValueError("pair_measurements must not be empty")

    sum_wxy = 0.0
    sum_wxx = 0.0
    for name_a, name_b, meas_a, meas_b in pair_measurements:
        if name_a not in profiles or name_b not in profiles:
            raise ValueError(f"pair ({name_a!r}, {name_b!r}) references profiles not in mapping")
        pa = profiles[name_a]
        pb = profiles[name_b]
        if pa.theta is None or pb.theta is None:
            raise ValueError(
                f"fit_gamma_kernel_size requires fitted theta on every profile; "
                f"got theta[{name_a}]={pa.theta}, theta[{name_b}]={pb.theta}"
            )
        if meas_a <= 0 or meas_b <= 0:
            raise ValueError(
                f"observed latencies must be > 0 for relative-residual fit; "
                f"got meas_a={meas_a}, meas_b={meas_b} for pair ({name_a!r}, {name_b!r})"
            )
        sum_act = pa.act_solo_ms + pb.act_solo_ms
        if sum_act == 0:
            continue
        sched_delay = hw.scheduling_delay_ms_at_concurrency(2)
        kd_a = pa.kernel_duration_ms
        kd_b = pb.kernel_duration_ms
        if kd_a == 0 or kd_b == 0:
            continue
        # Self = A
        frac_a = pb.act_solo_ms / sum_act
        ksr_a = kd_b / kd_a
        y_a = meas_a - pa.baseidle_ms - sched_delay - pa.act_solo_ms * (1.0 + pa.theta * frac_a)
        x_a = pa.act_solo_ms * pa.theta * frac_a * ksr_a
        w_a = 1.0 / (meas_a * meas_a)
        sum_wxy += w_a * x_a * y_a
        sum_wxx += w_a * x_a * x_a
        # Self = B
        frac_b = pa.act_solo_ms / sum_act
        ksr_b = kd_a / kd_b
        y_b = meas_b - pb.baseidle_ms - sched_delay - pb.act_solo_ms * (1.0 + pb.theta * frac_b)
        x_b = pb.act_solo_ms * pb.theta * frac_b * ksr_b
        w_b = 1.0 / (meas_b * meas_b)
        sum_wxy += w_b * x_b * y_b
        sum_wxx += w_b * x_b * x_b

    if sum_wxx == 0.0:
        raise ValueError(
            "fit_gamma_kernel_size regressor sum collapsed to zero; "
            "calibration data is degenerate (zero kernel_size_ratio variance)"
        )
    return sum_wxy / sum_wxx


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
) -> tuple[float, float] | None:
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
    pair_lookup: Mapping[tuple[str, str], PairLookupEntry] | None = None,
    gamma_kernel_size: float | None = None,
) -> tuple[float, float, str]:
    """Tiered pair latency prediction: lookup first, then queue-aware (V5 if γ supplied).

    Parameters
    ----------
    pair_lookup
        Per-pair measured-ground-truth safety net. If the pair is present,
        its measurement is returned directly (source = ``"lookup"``).
    gamma_kernel_size
        Optional V5 kernel-size-ratio coefficient. When supplied (non-None
        and non-zero), routes the queue-aware path through
        :func:`predict_pair_latency_queue_aware_v5`. SUBSTRATE-BOUND: the
        operator-fitted γ value is meaningful only for the substrate it was
        fitted on. Default ``None`` reproduces v0.2 queue-aware behaviour.

    Returns
    -------
    tuple
        ``(latency_a_ms, latency_b_ms, source)`` where source is one of:

        - ``"lookup"`` — measured ground truth from ``pair_lookup``.
        - ``"queue_aware_v5"`` — fitted V5 formula prediction (γ supplied).
        - ``"queue_aware"`` — fitted v0.2 queue-aware formula prediction.
        - ``"fallback_unknown"`` — both profiles lack theta and pair_lookup
          had no entry; returns standalone ``act_solo_ms`` (no interference).
    """
    pair = (profile_a.name, profile_b.name)
    if pair_lookup is not None:
        result = lookup_pair_latency(pair, pair_lookup)
        if result is not None:
            return result[0], result[1], "lookup"

    if profile_a.theta is not None and profile_b.theta is not None:
        if gamma_kernel_size is not None and gamma_kernel_size != 0.0:
            a, b = predict_pair_latency_queue_aware_v5(
                profile_a, profile_b, hw, gamma_kernel_size=gamma_kernel_size
            )
            return a, b, "queue_aware_v5"
        a, b = predict_pair_latency_queue_aware(profile_a, profile_b, hw)
        return a, b, "queue_aware"

    # No theta + no lookup: cannot predict; return standalone (no interference)
    return profile_a.act_solo_ms, profile_b.act_solo_ms, "fallback_unknown"
