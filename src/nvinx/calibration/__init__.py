"""nvinx.calibration — turnkey operator workflow for calibrating an nvinx bench.

This subpackage lifts what was previously a research-shape calibration toolkit
into a library API + CLI so operators can produce ``HardwareCoefficients`` and
per-model ``InterferenceProfile`` objects for their own substrate without
writing their own scripts.

Public API
----------

End-to-end orchestrator (the cheapest entry point)::

    from nvinx.calibration import run_calibration, CalibrationResult

    result: CalibrationResult = run_calibration(
        model_loaders={
            "model_a": lambda: load_my_model("checkpoint_a"),
            "model_b": lambda: load_my_model("checkpoint_b"),
            # ...
        },
        output_dir=Path("./calibration_2026-05-12"),
        fit_v5=True,  # also fit V5 gamma_kernel_size on top of theta
    )

    # result.hw_coefs                 → HardwareCoefficients
    # result.profiles                 → dict[str, InterferenceProfile] (theta fitted)
    # result.gamma_kernel_size        → float | None (V5)
    # result.lopo_mean_pct            → float
    # result.lopo_mean_pct_v5         → float | None

CLI (same orchestrator, command-line driven)::

    $ nvinx-calibrate \\
        --model my_pkg:loader_a \\
        --model my_pkg:loader_b \\
        --output ./calibration_2026-05-12 \\
        --fit-v5

Per-step API (for operators who want fine-grained control)::

    from nvinx.calibration import (
        sweep_hardware,      # idlef polynomial + powerp linear sweep
        profile_model,       # per-model standalone profiling (ncu L2 + act_solo)
        validate_pair,       # cross-pair co-located latency measurement
        fit_thetas,          # least-squares theta fitting on relative residuals
        fit_v5,              # joint theta + gamma_kernel_size fit
        lopo_cross_validate, # LOPO % error summary
    )

Optional dependencies
---------------------

This subpackage requires extras. Install with::

    pip install nvinx[calibration]

The extras pull in:

- ``scipy`` — for ``scipy.optimize.least_squares`` (joint theta fitting)
- ``numpy`` — array math for residuals
- ``nvidia-ml-py`` — for NVML power-draw telemetry during the hardware sweep

System prerequisites (not Python-installable):

- ``ncu`` (Nsight Compute) for L2 cache saturation measurement. Either run the
  calibration via ``sudo`` or set ``NVreg_RestrictProfilingToAdminUsers=0`` on
  the NVIDIA kernel module.
- NVIDIA driver + matching CUDA + a recent PyTorch (or the framework you use
  to load models).

Substrate-agnostic
------------------

This package queries the actual GPU via ``nvidia-smi`` for substrate constants
(no hardcoded mobile/datacenter values). Model loaders are operator-supplied
(no hardcoded model registry). The output ``CalibrationResult`` is your
substrate's calibration; ship it alongside your application's interference
profiles, or pin it to ``research/v0_2_calibration/`` if you only want to
reproduce the reference bench's numbers.

Honest scope
------------

The calibration produces empirical coefficients fitted on your bench's
measurement data. Predictions made with those coefficients will be accurate
on the same bench under the same driver / workload class. Cross-substrate
generalization is **not** automatic — the substrate-bound discipline in the
v0.3 ``nvinx.interference`` docstrings applies here too: if your bench
changes (different GPU, different model class, different driver), re-run
the calibration.

Lazy imports
------------

To keep ``import nvinx`` (and the runtime-only V5 prediction path) free of
scipy / numpy / nvidia-ml-py, this subpackage's heavy imports happen inside
function bodies. ``from nvinx.calibration import run_calibration`` works on
a Python install without the extras; calling ``run_calibration(...)`` then
raises a clear ImportError with the install instruction if extras are missing.
"""

from nvinx.calibration.fit import (
    apply_thetas,
    fit_thetas,
    fit_v5,
    lopo_cross_validate,
)
from nvinx.calibration.hardware import sweep_hardware
from nvinx.calibration.profile import ProfileTarget, profile_model
from nvinx.calibration.result import CalibrationResult
from nvinx.calibration.runner import run_calibration
from nvinx.calibration.validate import validate_pair

__all__ = [
    "CalibrationResult",
    "ProfileTarget",
    "apply_thetas",
    "fit_thetas",
    "fit_v5",
    "lopo_cross_validate",
    "profile_model",
    "run_calibration",
    "sweep_hardware",
    "validate_pair",
]
