"""CalibrationResult dataclass — end-to-end output of nvinx.calibration."""

from __future__ import annotations

from dataclasses import dataclass, field

from nvinx.interference import HardwareCoefficients, InterferenceProfile


@dataclass(frozen=True)
class CalibrationResult:
    """End-to-end output of :func:`nvinx.calibration.run_calibration`.

    Attributes
    ----------
    hw_coefs
        Substrate-level coefficients produced by :func:`sweep_hardware`.
    profiles
        Per-model :class:`nvinx.interference.InterferenceProfile` objects with
        ``theta`` fitted. Operator-generated; ship alongside the
        ``hw_coefs`` for runtime prediction.
    pair_measurements
        Cross-pair co-located latency measurements that drove the theta fit
        and (optionally) the V5 gamma fit. Each tuple is
        ``(name_a, name_b, meas_a_ms, meas_b_ms)``.
    gamma_kernel_size
        Optional V5 gamma_kernel_size coefficient if ``fit_v5=True`` was
        passed to :func:`run_calibration`. ``None`` if only V1 thetas were
        fitted.
    lopo_mean_pct
        Leave-one-pair-out cross-validation mean absolute % error on the
        V1 baseline queue-aware formula (theta-only fit).
    lopo_max_pct
        LOPO max % error (V1 baseline).
    lopo_mean_pct_v5
        LOPO mean % error on the V5 formula. ``None`` if V5 wasn't fitted.
    lopo_max_pct_v5
        LOPO max % error on the V5 formula. ``None`` if V5 wasn't fitted.
    notes
        Free-form list of strings describing the calibration run (substrate
        detected, ncu version, optimizer iterations, etc.).
    """

    hw_coefs: HardwareCoefficients
    profiles: dict[str, InterferenceProfile]
    pair_measurements: list[tuple[str, str, float, float]]
    lopo_mean_pct: float
    lopo_max_pct: float
    gamma_kernel_size: float | None = None
    lopo_mean_pct_v5: float | None = None
    lopo_max_pct_v5: float | None = None
    notes: list[str] = field(default_factory=list)
