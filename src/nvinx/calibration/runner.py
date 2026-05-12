"""End-to-end calibration orchestrator + ``nvinx-calibrate`` CLI.

Drives the full pipeline::

    sweep_hardware
        → for each model: profile_model (theta=None)
        → for each pair: load both → validate_pair → measurement tuple
        → fit_thetas (and optionally fit_v5)
        → lopo_cross_validate
        → CalibrationResult written to <output_dir>/calibration_result.json

Operator supplies a mapping ``{model_name: loader_callable}`` (or, in the
CLI, ``--model my_pkg:loader_fn`` arguments). Each loader is called
on-demand to produce a fresh :class:`ProfileTarget`; the orchestrator
manages VRAM lifecycle (release between profiles, re-load for each pair).
"""

from __future__ import annotations

import argparse
import importlib
import itertools
import json
import sys
from collections.abc import Callable, Mapping
from dataclasses import asdict
from pathlib import Path

from nvinx.calibration.fit import (
    apply_thetas,
    fit_thetas,
    fit_v5,
    lopo_cross_validate,
)
from nvinx.calibration.hardware import sweep_hardware
from nvinx.calibration.profile import ProfileTarget, profile_model
from nvinx.calibration.result import CalibrationResult
from nvinx.calibration.validate import validate_pair


def _release_vram() -> None:
    """Free Python references to GPU tensors + empty the CUDA cache."""
    import gc

    try:
        import torch

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except ImportError:
        gc.collect()


def run_calibration(
    model_loaders: Mapping[str, Callable[[], ProfileTarget]],
    *,
    output_dir: Path,
    fit_v5_gamma: bool = False,
    n_warmup: int = 3,
    n_measure: int = 5,
    ncu_sudo: bool = True,
    architecture_classes: Mapping[str, str] | None = None,
    skip_hardware_sweep: bool = False,
    hw_coefs_override=None,  # type: ignore[no-untyped-def]
) -> CalibrationResult:
    """Run a full nvinx calibration end-to-end on the current substrate.

    Parameters
    ----------
    model_loaders
        Mapping from model name to a zero-arg callable returning a fresh
        :class:`ProfileTarget`. At least 3 models recommended (so C(3,2) =
        3 pairs gives 6 measurements for theta fitting).
    output_dir
        Directory to dump per-model JSONs, per-pair JSONs, hardware
        constants, and the final :class:`CalibrationResult`. Created if
        absent.
    fit_v5_gamma
        Whether to additionally fit V5 ``gamma_kernel_size``. Default
        ``False`` — fit V1 theta only.
    n_warmup, n_measure
        Forwarded to :func:`profile_model` and :func:`validate_pair`.
    ncu_sudo
        Forwarded to :func:`profile_model` (sudo invocation of ncu).
    architecture_classes
        Optional ``{model_name: architecture_class}`` mapping (advisory
        tags like ``"encoder_transformer"``). Defaults to ``"unknown"``.
    skip_hardware_sweep
        If True, ``hw_coefs_override`` must be supplied and the hardware
        sweep is skipped. Useful when the substrate constants are already
        known and you only want to refresh per-model + per-pair data.
    hw_coefs_override
        :class:`HardwareCoefficients` to use when
        ``skip_hardware_sweep=True``.

    Returns
    -------
    CalibrationResult
        End-to-end result; also serialized to
        ``<output_dir>/calibration_result.json``.

    Raises
    ------
    ValueError
        If fewer than 3 model loaders are supplied, or if
        ``skip_hardware_sweep=True`` without ``hw_coefs_override``.
    """
    if len(model_loaders) < 3:
        raise ValueError(
            "run_calibration requires at least 3 model loaders for a meaningful"
            f" theta fit; got {len(model_loaders)}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    notes: list[str] = []

    if skip_hardware_sweep:
        if hw_coefs_override is None:
            raise ValueError("skip_hardware_sweep=True requires hw_coefs_override")
        hw = hw_coefs_override
        notes.append("Hardware sweep skipped; using operator-supplied HardwareCoefficients.")
    else:
        hw = sweep_hardware(output_dir=output_dir)
        notes.append(f"Hardware sweep complete: substrate={hw.substrate_name}")

    profiles: dict[
        str, object
    ] = {}  # InterferenceProfile, typed loose to satisfy mypy on later replace
    arch_map = dict(architecture_classes or {})
    for name, loader in model_loaders.items():
        profile = profile_model(
            loader,
            architecture_class=arch_map.get(name, "unknown"),
            n_warmup=n_warmup,
            n_measure=n_measure,
            ncu_sudo=ncu_sudo,
            output_dir=output_dir,
        )
        profiles[name] = profile
        _release_vram()
        notes.append(
            f"Profiled {name}: act_solo_ms={profile.act_solo_ms:.2f} "
            f"kernels={profile.kernels} l2_sat={profile.l2_saturation_pct:.1f}%"
        )

    pair_measurements: list[tuple[str, str, float, float]] = []
    for name_a, name_b in itertools.combinations(sorted(model_loaders.keys()), 2):
        target_a = model_loaders[name_a]()
        target_b = model_loaders[name_b]()
        try:
            measurement = validate_pair(
                target_a, target_b, n_warmup=max(2, n_warmup - 1), n_measure=n_measure
            )
            pair_measurements.append(measurement)
            (output_dir / f"pair_{name_a}_{name_b}.json").write_text(
                json.dumps(
                    {
                        "pair": [name_a, name_b],
                        "measured_latencies_ms": [measurement[2], measurement[3]],
                    },
                    indent=2,
                )
            )
            notes.append(f"Pair {name_a}+{name_b}: {measurement[2]:.2f}ms / {measurement[3]:.2f}ms")
        finally:
            del target_a, target_b
            _release_vram()

    thetas = fit_thetas(profiles, pair_measurements, hw)  # type: ignore[arg-type]
    profiles_fitted = apply_thetas(profiles, thetas)  # type: ignore[arg-type]
    lopo_v1 = lopo_cross_validate(profiles, pair_measurements, hw, refit_v5=False)  # type: ignore[arg-type]
    notes.append(f"V1 LOPO: mean={lopo_v1['mean_pct']:.2f}% max={lopo_v1['max_pct']:.2f}%")

    gamma: float | None = None
    lopo_mean_pct_v5: float | None = None
    lopo_max_pct_v5: float | None = None
    if fit_v5_gamma:
        _thetas_v5, gamma = fit_v5(profiles, pair_measurements, hw)  # type: ignore[arg-type]
        lopo_v5 = lopo_cross_validate(profiles, pair_measurements, hw, refit_v5=True)  # type: ignore[arg-type]
        lopo_mean_pct_v5 = lopo_v5["mean_pct"]
        lopo_max_pct_v5 = lopo_v5["max_pct"]
        notes.append(
            f"V5 LOPO (gamma={gamma:.4f}): mean={lopo_v5['mean_pct']:.2f}% "
            f"max={lopo_v5['max_pct']:.2f}%"
        )

    result = CalibrationResult(
        hw_coefs=hw,
        profiles=profiles_fitted,  # type: ignore[arg-type]
        pair_measurements=pair_measurements,
        lopo_mean_pct=lopo_v1["mean_pct"],
        lopo_max_pct=lopo_v1["max_pct"],
        gamma_kernel_size=gamma,
        lopo_mean_pct_v5=lopo_mean_pct_v5,
        lopo_max_pct_v5=lopo_max_pct_v5,
        notes=notes,
    )

    serializable = {
        "hw_coefs": {
            "idlef_polynomial": list(hw.idlef_polynomial),
            "powerp_linear": list(hw.powerp_linear),
            "nominal_freq_mhz": hw.nominal_freq_mhz,
            "tdp_watts": hw.tdp_watts,
            "substrate_name": hw.substrate_name,
        },
        "profiles": {n: asdict(p) for n, p in profiles_fitted.items()},  # type: ignore[arg-type]
        "pair_measurements": [list(m) for m in pair_measurements],
        "lopo_mean_pct": result.lopo_mean_pct,
        "lopo_max_pct": result.lopo_max_pct,
        "gamma_kernel_size": result.gamma_kernel_size,
        "lopo_mean_pct_v5": result.lopo_mean_pct_v5,
        "lopo_max_pct_v5": result.lopo_max_pct_v5,
        "notes": result.notes,
    }
    (output_dir / "calibration_result.json").write_text(json.dumps(serializable, indent=2))
    return result


def _resolve_loader(spec: str) -> Callable[[], ProfileTarget]:
    """Resolve a ``module:attr`` spec to a callable.

    Example: ``my_pkg.models:load_esm2`` imports ``my_pkg.models`` and
    returns its ``load_esm2`` attribute.
    """
    if ":" not in spec:
        raise ValueError(f"loader spec {spec!r} must be of the form 'module.path:attr_name'")
    module_path, attr = spec.rsplit(":", 1)
    module = importlib.import_module(module_path)
    loader = getattr(module, attr)
    if not callable(loader):
        raise ValueError(f"loader {spec!r} is not callable")
    return loader


def cli_main(argv: list[str] | None = None) -> int:
    """``nvinx-calibrate`` CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="nvinx-calibrate",
        description=(
            "Calibrate nvinx HardwareCoefficients + per-model "
            "InterferenceProfiles for the current substrate."
        ),
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help="Loader spec 'module.path:attr_name'. Repeatable; ≥3 required.",
    )
    parser.add_argument(
        "--name",
        action="append",
        default=None,
        help=(
            "Optional name for each --model. Repeatable; must match --model "
            "count if supplied. Defaults to the attr name."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for per-model + per-pair JSONs and CalibrationResult.",
    )
    parser.add_argument(
        "--fit-v5",
        action="store_true",
        help="Also fit V5 gamma_kernel_size in addition to V1 theta.",
    )
    parser.add_argument(
        "--no-sudo",
        action="store_true",
        help="Do not invoke ncu via sudo (requires NVreg_RestrictProfilingToAdminUsers=0).",
    )
    parser.add_argument("--n-warmup", type=int, default=3)
    parser.add_argument("--n-measure", type=int, default=5)
    args = parser.parse_args(argv)

    if args.name and len(args.name) != len(args.model):
        parser.error("--name count must match --model count when provided")

    loaders: dict[str, Callable[[], ProfileTarget]] = {}
    for i, spec in enumerate(args.model):
        loader = _resolve_loader(spec)
        name = args.name[i] if args.name else spec.rsplit(":", 1)[-1]
        loaders[name] = loader

    if len(loaders) < 3:
        parser.error("at least 3 distinct --model specs required for a meaningful fit")

    result = run_calibration(
        loaders,
        output_dir=args.output,
        fit_v5_gamma=args.fit_v5,
        n_warmup=args.n_warmup,
        n_measure=args.n_measure,
        ncu_sudo=not args.no_sudo,
    )
    print(f"V1 LOPO: mean={result.lopo_mean_pct:.2f}% max={result.lopo_max_pct:.2f}%")
    if result.gamma_kernel_size is not None:
        print(
            f"V5 gamma={result.gamma_kernel_size:.4f}; "
            f"LOPO mean={result.lopo_mean_pct_v5:.2f}% max={result.lopo_max_pct_v5:.2f}%"
        )
    print(f"CalibrationResult written to {args.output / 'calibration_result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(cli_main())
