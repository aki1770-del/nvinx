"""Tests for nvinx.calibration.fit.

Pure-math tests (no GPU, no real ncu needed). Verify:

- fit_thetas recovers known thetas from synthetic V1 measurements.
- fit_v5 recovers known (thetas, gamma) from synthetic V5 measurements.
- apply_thetas applies fitted theta back onto the profile dict.
- lopo_cross_validate produces non-negative mean / max consistent with
  fit.
- Error paths (empty measurements, unknown profile names).

Tests are gated on ``scipy`` availability — install ``[calibration]``
extras to run them; otherwise the module is skipped entirely.
"""

from __future__ import annotations

import pytest

pytest.importorskip("scipy")
pytest.importorskip("numpy")

from nvinx.calibration.fit import (  # noqa: E402
    apply_thetas,
    fit_thetas,
    fit_v5,
    lopo_cross_validate,
)
from nvinx.interference import (  # noqa: E402
    HardwareCoefficients,
    InterferenceProfile,
    predict_pair_latency_queue_aware,
    predict_pair_latency_queue_aware_v5,
)

HW = HardwareCoefficients(
    idlef_polynomial=(6.42, -7.0),
    powerp_linear=(0.0,),
    nominal_freq_mhz=1530.0,
    tdp_watts=40.0,
    substrate_name="rtx_a1000_4gb",
)


def _make_profile(name: str, kernels: int, act_solo_ms: float, theta: float | None = None):
    return InterferenceProfile(
        name=name,
        kernels=kernels,
        baseidle_ms=0.0,
        act_solo_ms=act_solo_ms,
        l2_saturation_pct=0.0,
        theta=theta,
    )


def _synthesize_v1_measurements(profiles, hw):
    """Generate noise-free V1 queue-aware measurements for all pairs."""
    names = sorted(profiles.keys())
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pa, pb = profiles[a], profiles[b]
            lat_a, lat_b = predict_pair_latency_queue_aware(pa, pb, hw)
            pairs.append((a, b, lat_a, lat_b))
    return pairs


def _synthesize_v5_measurements(profiles, hw, gamma):
    """Generate noise-free V5 measurements for all pairs."""
    names = sorted(profiles.keys())
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pa, pb = profiles[a], profiles[b]
            lat_a, lat_b = predict_pair_latency_queue_aware_v5(pa, pb, hw, gamma_kernel_size=gamma)
            pairs.append((a, b, lat_a, lat_b))
    return pairs


def test_fit_thetas_recovers_known_thetas_noise_free():
    known = {
        "long_kernel_model": 1.20,
        "mid_kernel_model": 2.50,
        "short_kernel_model": 4.10,
    }
    profiles_with_known = {
        "long_kernel_model": _make_profile(
            "long_kernel_model", 1000, 100.0, theta=known["long_kernel_model"]
        ),
        "mid_kernel_model": _make_profile(
            "mid_kernel_model", 800, 50.0, theta=known["mid_kernel_model"]
        ),
        "short_kernel_model": _make_profile(
            "short_kernel_model", 600, 20.0, theta=known["short_kernel_model"]
        ),
    }
    pairs = _synthesize_v1_measurements(profiles_with_known, HW)
    # Hand fitter profiles WITHOUT theta — it should still fit
    profiles_no_theta = {
        n: _make_profile(p.name, p.kernels, p.act_solo_ms, theta=None)
        for n, p in profiles_with_known.items()
    }
    fitted = fit_thetas(profiles_no_theta, pairs, HW)
    for name, true_theta in known.items():
        assert fitted[name] == pytest.approx(true_theta, rel=1e-3)


def test_fit_v5_recovers_known_thetas_and_gamma_noise_free():
    known_gamma = 0.75
    known_thetas = {
        "long_kernel_model": 1.20,
        "mid_kernel_model": 2.50,
        "short_kernel_model": 4.10,
    }
    profiles = {
        n: _make_profile(n, kerns, act, theta=t)
        for n, t, kerns, act in [
            ("long_kernel_model", 1.20, 1000, 100.0),
            ("mid_kernel_model", 2.50, 800, 50.0),
            ("short_kernel_model", 4.10, 600, 20.0),
        ]
    }
    pairs = _synthesize_v5_measurements(profiles, HW, known_gamma)
    fitted_thetas, fitted_gamma = fit_v5(profiles, pairs, HW)
    assert fitted_gamma == pytest.approx(known_gamma, rel=1e-3)
    for name, true_theta in known_thetas.items():
        assert fitted_thetas[name] == pytest.approx(true_theta, rel=1e-3)


def test_apply_thetas_sets_theta_on_each_profile():
    profiles = {
        "a": _make_profile("a", 100, 10.0, theta=None),
        "b": _make_profile("b", 200, 20.0, theta=None),
    }
    fitted = {"a": 1.5, "b": 2.5}
    out = apply_thetas(profiles, fitted)
    assert out["a"].theta == 1.5
    assert out["b"].theta == 2.5
    # Originals are frozen — unchanged
    assert profiles["a"].theta is None
    assert profiles["b"].theta is None


def test_fit_thetas_empty_pair_measurements_raises():
    profiles = {"a": _make_profile("a", 100, 10.0)}
    with pytest.raises(ValueError, match="must not be empty"):
        fit_thetas(profiles, [], HW)


def test_fit_thetas_unknown_profile_raises():
    profiles = {"a": _make_profile("a", 100, 10.0)}
    pairs = [("a", "unknown", 1.0, 1.0)]
    with pytest.raises(ValueError, match="references profiles not in mapping"):
        fit_thetas(profiles, pairs, HW)


def test_lopo_cross_validate_smoke():
    profiles = {
        n: _make_profile(n, kerns, act, theta=1.0)
        for n, kerns, act in [
            ("a", 100, 10.0),
            ("b", 200, 20.0),
            ("c", 400, 40.0),
        ]
    }
    pairs = _synthesize_v1_measurements(profiles, HW)
    summary = lopo_cross_validate(profiles, pairs, HW, refit_v5=False)
    assert summary["n_pairs"] == 3
    assert summary["n_observations"] == 6
    # Noise-free V1 measurements + correct V1 predictor → LOPO error near zero
    assert summary["mean_pct"] < 0.5
    assert summary["max_pct"] < 1.0


def test_lopo_v5_refit_path():
    gamma = 0.5
    profiles = {
        n: _make_profile(n, kerns, act, theta=1.0)
        for n, kerns, act in [
            ("a", 100, 10.0),
            ("b", 200, 20.0),
            ("c", 400, 40.0),
        ]
    }
    pairs = _synthesize_v5_measurements(profiles, HW, gamma)
    summary = lopo_cross_validate(profiles, pairs, HW, refit_v5=True)
    assert summary["n_pairs"] == 3
    # Noise-free V5 measurements + correct V5 predictor → LOPO error near zero
    assert summary["mean_pct"] < 0.5
