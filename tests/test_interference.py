"""Tests for v0.2/v0.3 interference primitives.

Validates the queue-aware substrate-native interference model that emerged
from the v0.2-v0.4 calibration journey on RTX A1000 mobile + heterogeneous
transformer/LLM corpus. LOPO mean error ~16% on 4-model corpus.
"""

import pytest

from nvinx.catalog import HardwareSpec, ModelSpec, Residency
from nvinx.interference import (
    HardwareCoefficients,
    InterferenceProfile,
    PairLookupEntry,
    asymmetry_predictor,
    lookup_pair_latency,
    max_kernel_rate_score,
    predict_pair_latency,
    predict_pair_latency_queue_aware,
)
from nvinx.patterns import fractional_coresidency, fractional_coresidency_v2

HW_4GB = HardwareSpec(vram_gb=4.0, ram_gb=32.0, cpu_cores=8)
HW_COEFS = HardwareCoefficients(
    idlef_polynomial=(6.42, -7.0),  # representative RTX A1000 mobile values
    powerp_linear=(0.0,),
    nominal_freq_mhz=1530.0,
    tdp_watts=40.0,
    substrate_name="rtx_a1000_4gb",
)

# Profiles drawn from v0.4 5-model corpus (RTX A1000 mobile bench)
PROF_LONG = InterferenceProfile(
    name="esm2_long",
    kernels=1026,
    baseidle_ms=0.149,
    act_solo_ms=114.9,
    l2_saturation_pct=37.1,
    theta=1.20,
    architecture_class="encoder_transformer",
)
PROF_SHORT = InterferenceProfile(
    name="esm2_short",
    kernels=1027,
    baseidle_ms=0.077,
    act_solo_ms=21.7,
    l2_saturation_pct=16.8,
    theta=3.91,
    architecture_class="encoder_transformer",
)
PROF_QWEN = InterferenceProfile(
    name="qwen_05b",
    kernels=1005,
    baseidle_ms=0.022,
    act_solo_ms=12.5,
    l2_saturation_pct=5.2,
    theta=5.64,
    architecture_class="decoder_transformer",
)
PROF_WHISPER = InterferenceProfile(
    name="whisper_base",
    kernels=335,
    baseidle_ms=0.033,
    act_solo_ms=58.3,
    l2_saturation_pct=42.0,
    theta=0.92,
    architecture_class="encoder_decoder",
)


# ============================================================================
# Profile property tests
# ============================================================================


def test_kernel_rate_property():
    assert PROF_LONG.kernel_rate == pytest.approx(1026 / 114.9, rel=0.01)
    assert PROF_SHORT.kernel_rate == pytest.approx(1027 / 21.7, rel=0.01)


def test_kernel_duration_property():
    # Long has bigger kernels than short
    assert PROF_LONG.kernel_duration_ms > PROF_SHORT.kernel_duration_ms
    # Whisper has biggest kernels
    assert PROF_WHISPER.kernel_duration_ms > PROF_LONG.kernel_duration_ms


def test_kernel_rate_zero_safety():
    p = InterferenceProfile(
        name="empty", kernels=0, baseidle_ms=0, act_solo_ms=0, l2_saturation_pct=0
    )
    assert p.kernel_rate == 0.0
    assert p.kernel_duration_ms == 0.0


# ============================================================================
# Hardware coefficients
# ============================================================================


def test_scheduling_delay_zero_at_n_concurrent_one():
    assert HW_COEFS.scheduling_delay_ms_at_concurrency(1) == 0.0


def test_scheduling_delay_positive_at_n_concurrent_two():
    delay = HW_COEFS.scheduling_delay_ms_at_concurrency(2)
    assert delay > 0


# ============================================================================
# Queue-aware prediction
# ============================================================================


def test_queue_aware_predict_long_qwen_pair():
    """Long+Qwen pair from v0.4: long predicted close to standalone, qwen suffers."""
    lat_long, lat_qwen = predict_pair_latency_queue_aware(PROF_LONG, PROF_QWEN, HW_COEFS)
    # Long's act_solo_ms is 114.9; co-located it should be only modestly slower
    assert lat_long > PROF_LONG.act_solo_ms
    assert lat_long < PROF_LONG.act_solo_ms * 2.0
    # Qwen's act_solo_ms is 12.5; co-located with big-partner long it should suffer
    assert lat_qwen > PROF_QWEN.act_solo_ms * 3.0


def test_queue_aware_requires_theta():
    p = InterferenceProfile(
        name="x", kernels=100, baseidle_ms=0.0, act_solo_ms=10.0, l2_saturation_pct=10.0,
        theta=None,  # missing
    )
    with pytest.raises(ValueError, match="theta"):
        predict_pair_latency_queue_aware(p, PROF_LONG, HW_COEFS)


def test_queue_aware_zero_act_solo_safety():
    p = InterferenceProfile(
        name="x", kernels=0, baseidle_ms=0.0, act_solo_ms=0.0, l2_saturation_pct=0.0,
        theta=1.0,
    )
    a, b = predict_pair_latency_queue_aware(p, p, HW_COEFS)
    assert a == 0.0 and b == 0.0


# ============================================================================
# Heuristic augmentations
# ============================================================================


def test_max_kernel_rate_score():
    """Qwen has highest kernel rate; placement containing it should reflect that."""
    score = max_kernel_rate_score([PROF_LONG, PROF_QWEN])
    assert score == pytest.approx(PROF_QWEN.kernel_rate, rel=0.01)
    score_no_qwen = max_kernel_rate_score([PROF_LONG, PROF_WHISPER])
    assert score_no_qwen < score


def test_max_kernel_rate_empty():
    assert max_kernel_rate_score([]) == 0.0


def test_asymmetry_predictor():
    """Long+qwen pair has high asymmetry; long+whisper less."""
    asym_long_qwen = asymmetry_predictor(PROF_LONG, PROF_QWEN)
    asym_long_whisper = asymmetry_predictor(PROF_LONG, PROF_WHISPER)
    assert asym_long_qwen > asym_long_whisper


def test_asymmetry_predictor_symmetric():
    """Order shouldn't matter."""
    a = asymmetry_predictor(PROF_LONG, PROF_QWEN)
    b = asymmetry_predictor(PROF_QWEN, PROF_LONG)
    assert a == b


# ============================================================================
# Pair lookup safety net
# ============================================================================


def test_lookup_pair_latency_hit():
    entry = PairLookupEntry(
        pair=("esm2_short", "qwen_05b"),
        measured_latency_a_ms=76.5,
        measured_latency_b_ms=77.9,
    )
    lookup = {("esm2_short", "qwen_05b"): entry}  # canonical order
    result = lookup_pair_latency(("esm2_short", "qwen_05b"), lookup)
    assert result == (76.5, 77.9)
    # Reversed order should also work
    result2 = lookup_pair_latency(("qwen_05b", "esm2_short"), lookup)
    assert result2 == (77.9, 76.5)


def test_lookup_pair_latency_miss():
    lookup = {}
    assert lookup_pair_latency(("a", "b"), lookup) is None


# ============================================================================
# Tiered prediction (predict_pair_latency)
# ============================================================================


def test_predict_pair_latency_lookup_first():
    entry = PairLookupEntry(
        pair=("esm2_short", "qwen_05b"),
        measured_latency_a_ms=76.5,
        measured_latency_b_ms=77.9,
    )
    canonical = tuple(sorted(["esm2_short", "qwen_05b"]))
    lookup = {canonical: entry}
    a, b, source = predict_pair_latency(PROF_SHORT, PROF_QWEN, HW_COEFS, pair_lookup=lookup)
    assert source == "lookup"
    # Should be 76.5/77.9 in some order
    assert {a, b} == {76.5, 77.9}


def test_predict_pair_latency_falls_back_to_queue_aware():
    a, b, source = predict_pair_latency(PROF_LONG, PROF_QWEN, HW_COEFS, pair_lookup=None)
    assert source == "queue_aware"
    assert a > 0 and b > 0


def test_predict_pair_latency_fallback_unknown_when_no_theta_no_lookup():
    p_no_theta = InterferenceProfile(
        name="x", kernels=100, baseidle_ms=0, act_solo_ms=10.0, l2_saturation_pct=10.0,
        theta=None,
    )
    a, b, source = predict_pair_latency(p_no_theta, PROF_LONG, HW_COEFS, pair_lookup=None)
    assert source == "fallback_unknown"
    assert a == p_no_theta.act_solo_ms
    assert b == PROF_LONG.act_solo_ms


# ============================================================================
# fractional_coresidency_v2 (backward-compat + augmentations)
# ============================================================================


SHORT_SPEC = ModelSpec(name="esm2_short", vram_gb=0.6, residency=Residency.GPU_SHARED)
LONG_SPEC = ModelSpec(name="esm2_long", vram_gb=0.6, residency=Residency.GPU_SHARED)
QWEN_SPEC = ModelSpec(name="qwen_05b", vram_gb=1.0, residency=Residency.GPU_SHARED)


def test_v2_without_profiles_equals_v1():
    """No interference_profiles → identical placement to v0.1."""
    candidates = [SHORT_SPEC, LONG_SPEC, QWEN_SPEC]
    plan_v1 = fractional_coresidency(candidates, HW_4GB)
    plan_v2 = fractional_coresidency_v2(candidates, HW_4GB)
    assert [m.name for m in plan_v1.gpu_coresident] == [m.name for m in plan_v2.gpu_coresident]
    assert plan_v1.cpu_parallel == plan_v2.cpu_parallel
    assert plan_v1.unscheduled == plan_v2.unscheduled


def test_v2_with_profiles_adds_diagnostic_notes():
    """interference_profiles → notes augmented with predictions."""
    candidates = [SHORT_SPEC, LONG_SPEC]
    profiles = {
        "esm2_short": PROF_SHORT,
        "esm2_long": PROF_LONG,
    }
    plan = fractional_coresidency_v2(
        candidates, HW_4GB, interference_profiles=profiles, hw_coefs=HW_COEFS
    )
    # Notes should contain interference info
    interference_notes = [n for n in plan.notes if "interference" in n]
    assert len(interference_notes) > 0
    # Should have a max_kernel_rate note
    mkr_notes = [n for n in interference_notes if "max_kernel_rate" in n]
    assert len(mkr_notes) > 0
    # Should have a pair prediction note
    pair_notes = [n for n in interference_notes if "pair(" in n]
    assert len(pair_notes) > 0


def test_v2_with_threshold_emits_warning():
    candidates = [SHORT_SPEC, QWEN_SPEC]
    profiles = {"esm2_short": PROF_SHORT, "qwen_05b": PROF_QWEN}
    plan = fractional_coresidency_v2(
        candidates, HW_4GB,
        interference_profiles=profiles,
        hw_coefs=HW_COEFS,
        max_kernel_rate_threshold=10.0,  # very low; will be exceeded
    )
    warning_notes = [n for n in plan.notes if "WARNING" in n]
    assert len(warning_notes) > 0


def test_v2_with_lookup_uses_lookup_source():
    candidates = [SHORT_SPEC, QWEN_SPEC]
    profiles = {"esm2_short": PROF_SHORT, "qwen_05b": PROF_QWEN}
    canonical = tuple(sorted(["esm2_short", "qwen_05b"]))
    lookup = {canonical: PairLookupEntry(
        pair=canonical,
        measured_latency_a_ms=76.5,
        measured_latency_b_ms=77.9,
    )}
    plan = fractional_coresidency_v2(
        candidates, HW_4GB,
        interference_profiles=profiles,
        hw_coefs=HW_COEFS,
        pair_lookup=lookup,
    )
    lookup_notes = [n for n in plan.notes if "via lookup" in n]
    assert len(lookup_notes) > 0
