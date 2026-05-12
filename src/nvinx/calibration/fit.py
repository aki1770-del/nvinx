"""Theta fitting + V5 gamma fitting + LOPO cross-validation.

The fit minimises **relative residuals** ``((pred − observed) / observed)²``
via ``scipy.optimize.least_squares`` with non-negativity bounds on theta.
This is the residual-norm convention an operator gets from
``scipy.optimize.least_squares`` on relative residuals — the appropriate norm
for substrates with wide ``act_solo`` spread across the corpus.

V5 gamma can be fitted jointly with thetas (:func:`fit_v5`) or — once thetas
are fitted under V1 — fitted alone via the closed-form weighted-LS in
:func:`nvinx.interference.fit_gamma_kernel_size`. Joint refit produces the
formula's minimum-residual operating point; closed-form refit produces a γ
consistent with the supplied thetas and is faster.

LOPO cross-validation reports honest generalization error: for each pair
in turn, hold it out, refit on the remaining N-1 pairs, predict the held-out
pair, record absolute % error per side; aggregate mean / max across all
2 × N held-out predictions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from nvinx.interference import (
    HardwareCoefficients,
    InterferenceProfile,
    predict_pair_latency_queue_aware,
    predict_pair_latency_queue_aware_v5,
)


def _require_scipy():
    try:
        import numpy as np  # noqa: F401
        from scipy.optimize import least_squares  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "nvinx.calibration.fit requires scipy + numpy. "
            "Install via: pip install nvinx[calibration]"
        ) from e


def _observations_from_pairs(
    profiles: Mapping[str, InterferenceProfile],
    pair_measurements: list[tuple[str, str, float, float]],
) -> list[tuple[int, int, float, float, int, int, float]]:
    """Flatten pair measurements to ``(self_idx, partner_idx, act_self, act_partner,
    k_self, k_partner, observed)`` rows; both directions of each pair contribute.
    """
    model_names = sorted(profiles.keys())
    name_to_idx = {n: i for i, n in enumerate(model_names)}
    rows: list[tuple[int, int, float, float, int, int, float]] = []
    for name_a, name_b, meas_a, meas_b in pair_measurements:
        if name_a not in name_to_idx or name_b not in name_to_idx:
            raise ValueError(f"pair ({name_a!r}, {name_b!r}) references profiles not in mapping")
        pa = profiles[name_a]
        pb = profiles[name_b]
        rows.append(
            (
                name_to_idx[name_a],
                name_to_idx[name_b],
                pa.act_solo_ms,
                pb.act_solo_ms,
                pa.kernels,
                pb.kernels,
                meas_a,
            )
        )
        rows.append(
            (
                name_to_idx[name_b],
                name_to_idx[name_a],
                pb.act_solo_ms,
                pa.act_solo_ms,
                pb.kernels,
                pa.kernels,
                meas_b,
            )
        )
    return rows


def fit_thetas(
    profiles: Mapping[str, InterferenceProfile],
    pair_measurements: list[tuple[str, str, float, float]],
    hw: HardwareCoefficients,
) -> dict[str, float]:
    """Fit per-model ``theta`` via ``scipy.optimize.least_squares`` on relative
    residuals.

    Parameters
    ----------
    profiles
        Mapping from model name to :class:`InterferenceProfile`. ``theta``
        values on input profiles are ignored (this function fits them);
        other fields (``act_solo_ms``, ``kernels``, ``baseidle_ms``) drive
        the prediction inside the residual function.
    pair_measurements
        ``(name_a, name_b, observed_self_ms, observed_partner_ms)`` per pair.
    hw
        Substrate :class:`HardwareCoefficients` (idlef polynomial drives the
        scheduling-delay term in the prediction).

    Returns
    -------
    dict
        ``{model_name: theta_fitted}``. To get profiles with fitted thetas,
        use :func:`apply_thetas`.

    Raises
    ------
    ImportError
        If scipy is not installed (extras ``[calibration]`` required).
    ValueError
        If ``pair_measurements`` is empty or references unknown profiles.
    """
    if not pair_measurements:
        raise ValueError("pair_measurements must not be empty")
    _require_scipy()
    import numpy as np
    from scipy.optimize import least_squares

    model_names = sorted(profiles.keys())
    n_models = len(model_names)
    name_to_idx = {n: i for i, n in enumerate(model_names)}
    observations = _observations_from_pairs(profiles, pair_measurements)

    sched_delay = hw.scheduling_delay_ms_at_concurrency(2)

    def residuals(theta_vec):
        thetas = np.maximum(theta_vec, 0.0)
        res = []
        for (
            self_i,
            _partner_i,
            act_self,
            act_partner,
            _k_self,
            _k_partner,
            observed,
        ) in observations:
            sum_act = act_self + act_partner
            if sum_act == 0:
                res.append(0.0)
                continue
            frac = act_partner / sum_act
            theta_self = float(thetas[self_i])
            baseidle_self = profiles[model_names[self_i]].baseidle_ms
            pred = baseidle_self + sched_delay + act_self * (1.0 + theta_self * frac)
            if observed == 0:
                res.append(0.0)
            else:
                res.append((pred - observed) / observed)
        return np.array(res)

    theta0 = np.ones(n_models)
    result = least_squares(residuals, theta0, bounds=(0.0, np.inf), max_nfev=2000)
    return {name: float(result.x[name_to_idx[name]]) for name in model_names}


def apply_thetas(
    profiles: Mapping[str, InterferenceProfile],
    thetas: Mapping[str, float],
) -> dict[str, InterferenceProfile]:
    """Return a new dict of profiles with theta fields set from ``thetas``.

    Input profiles are frozen; this is the canonical way to apply a fitted
    theta map back onto them.
    """
    return {name: replace(profile, theta=float(thetas[name])) for name, profile in profiles.items()}


def fit_v5(
    profiles: Mapping[str, InterferenceProfile],
    pair_measurements: list[tuple[str, str, float, float]],
    hw: HardwareCoefficients,
) -> tuple[dict[str, float], float]:
    """Joint-fit per-model ``theta`` + V5 ``gamma_kernel_size`` on relative
    residuals.

    Returns
    -------
    tuple
        ``(thetas, gamma)``. Both are substrate-bound.

    Raises
    ------
    ImportError
        If scipy is not installed.
    ValueError
        If ``pair_measurements`` is empty or references unknown profiles.
    """
    if not pair_measurements:
        raise ValueError("pair_measurements must not be empty")
    _require_scipy()
    import numpy as np
    from scipy.optimize import least_squares

    EPS = 1e-9
    model_names = sorted(profiles.keys())
    n_models = len(model_names)
    name_to_idx = {n: i for i, n in enumerate(model_names)}
    observations = _observations_from_pairs(profiles, pair_measurements)

    sched_delay = hw.scheduling_delay_ms_at_concurrency(2)

    def residuals(x):
        thetas = np.maximum(x[:n_models], 0.0)
        gamma = max(x[n_models], 0.0)
        res = []
        for self_i, _partner_i, act_self, act_partner, k_self, k_partner, observed in observations:
            sum_act = act_self + act_partner
            if sum_act == 0 or observed == 0:
                res.append(0.0)
                continue
            frac = act_partner / sum_act
            theta_self = float(thetas[self_i])
            baseidle_self = profiles[model_names[self_i]].baseidle_ms
            dur_self = act_self / k_self if k_self > 0 else EPS
            dur_partner = act_partner / k_partner if k_partner > 0 else EPS
            ksr = dur_partner / max(dur_self, EPS)
            v5_factor = 1.0 + gamma * ksr
            pred = baseidle_self + sched_delay + act_self * (1.0 + theta_self * frac * v5_factor)
            res.append((pred - observed) / observed)
        return np.array(res)

    x0 = np.concatenate([np.ones(n_models), np.array([0.5])])
    result = least_squares(residuals, x0, bounds=(0.0, np.inf), max_nfev=2000)
    thetas = {name: float(result.x[name_to_idx[name]]) for name in model_names}
    gamma = float(result.x[n_models])
    return thetas, gamma


def lopo_cross_validate(
    profiles: Mapping[str, InterferenceProfile],
    pair_measurements: list[tuple[str, str, float, float]],
    hw: HardwareCoefficients,
    *,
    refit_v5: bool = False,
) -> dict[str, float]:
    """Leave-one-pair-out cross-validation summary.

    For each pair held out: refit on the remaining N-1 pairs (theta-only if
    ``refit_v5=False``; joint theta+gamma if ``refit_v5=True``), predict the
    held-out pair using the freshly-fitted parameters, record absolute %
    error per side. Aggregate mean / max across all 2 × N held-out
    predictions.

    Parameters
    ----------
    profiles, pair_measurements, hw
        Same as :func:`fit_thetas`.
    refit_v5
        If True, joint-refit theta + gamma_kernel_size at each LOPO
        iteration (matches the v0.5 H5 joint-fit residual convention). If
        False, refit theta only and predict via the v0.2 queue-aware
        formula.

    Returns
    -------
    dict
        ``{"mean_pct": float, "max_pct": float, "n_pairs": int,
        "n_observations": int}``.

    Raises
    ------
    ImportError
        If scipy is not installed.
    """
    if len(pair_measurements) < 2:
        raise ValueError("LOPO requires at least 2 pair measurements")
    _require_scipy()

    abs_pct_errs: list[float] = []
    for held_idx, held in enumerate(pair_measurements):
        name_a, name_b, meas_a, meas_b = held
        training = [p for i, p in enumerate(pair_measurements) if i != held_idx]

        if refit_v5:
            thetas, gamma = fit_v5(profiles, training, hw)
        else:
            thetas = fit_thetas(profiles, training, hw)
            gamma = None

        pa = profiles[name_a]
        pb = profiles[name_b]
        pa_with_theta = replace(pa, theta=thetas[name_a])
        pb_with_theta = replace(pb, theta=thetas[name_b])
        if gamma is not None and gamma > 0.0:
            pred_a, pred_b = predict_pair_latency_queue_aware_v5(
                pa_with_theta, pb_with_theta, hw, gamma_kernel_size=gamma
            )
        else:
            pred_a, pred_b = predict_pair_latency_queue_aware(pa_with_theta, pb_with_theta, hw)
        abs_pct_errs.append(abs(pred_a - meas_a) / meas_a * 100.0)
        abs_pct_errs.append(abs(pred_b - meas_b) / meas_b * 100.0)

    return {
        "mean_pct": sum(abs_pct_errs) / len(abs_pct_errs),
        "max_pct": max(abs_pct_errs),
        "n_pairs": len(pair_measurements),
        "n_observations": len(abs_pct_errs),
    }
