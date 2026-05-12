"""Hardware sweep — idlef polynomial + powerp linear fit (substrate-agnostic).

Queries the actual GPU via ``nvidia-smi`` for substrate constants — no
hardcoded mobile / datacenter values. Two sweeps:

1. **idlef polynomial.** Per-inference scheduling delay added at N concurrent
   workloads. Default micro-benchmark is a small matmul launched on N
   independent CUDA streams; the per-stream median latency at each N feeds
   a degree-1 polynomial fit.

2. **powerp linear.** GPU frequency reduction slope (MHz / W) above TDP.
   Drives the GPU above TDP via sustained heavy matmul; samples
   ``(power.draw, clocks.current.graphics)`` via ``nvidia-smi``; fits a
   linear slope.

Both sweeps produce coefficients that operator's
:class:`nvinx.interference.HardwareCoefficients` consumes.

Honest scope
------------

The fitted coefficients are valid for the operator's actual substrate at
the moment of calibration. They drift with driver upgrades, thermal state
(laptop vs docked vs cooled), and large-context system-RAM contention.
Re-fit at material changes.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from nvinx.interference import HardwareCoefficients


def _require_runtime():
    try:
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "nvinx.calibration.hardware requires numpy + torch. "
            "Install scipy + numpy via: pip install nvinx[calibration] "
            "and install torch matching your CUDA via: "
            "https://pytorch.org/get-started/locally/"
        ) from e


def _query_nvidia_smi() -> dict[str, float | str]:
    """Query ``nvidia-smi`` for current GPU's name / clock / TDP.

    Returns a dict with keys ``substrate_name`` (str),
    ``nominal_freq_mhz`` (float), ``tdp_watts`` (float).

    Raises
    ------
    RuntimeError
        If ``nvidia-smi`` is not available or returns no GPU.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,clocks.max.graphics,power.max_limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        raise RuntimeError(
            "nvidia-smi query failed; is the NVIDIA driver installed and a GPU present?"
        ) from e

    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("nvidia-smi returned no GPU rows")

    parts = [p.strip() for p in lines[0].split(",")]
    if len(parts) < 3:
        raise RuntimeError(f"nvidia-smi parse failure: {lines[0]!r}")
    name, max_clock_mhz_str, tdp_watts_str = parts[0], parts[1], parts[2]
    return {
        "substrate_name": name.replace(" ", "_").lower(),
        "nominal_freq_mhz": float(max_clock_mhz_str),
        "tdp_watts": float(tdp_watts_str),
    }


def _matmul_kernel(size: int = 512, n_iter: int = 50, stream=None) -> float:
    """Micro-benchmark kernel for the idlef sweep. Returns elapsed ms."""
    import torch

    device = torch.device("cuda")
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    if stream is not None:
        with torch.cuda.stream(stream):
            start.record(stream)
            for _ in range(n_iter):
                _ = a @ b
            end.record(stream)
        stream.synchronize()
    else:
        start.record()
        for _ in range(n_iter):
            _ = a @ b
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end)


def _measure_idlef_polynomial(
    n_max: int = 5, n_trials: int = 5
) -> tuple[tuple[float, float], dict[int, float]]:
    """Idlef sweep: measure per-inference scheduling delay vs N concurrent.

    Returns ``((slope, intercept), raw_data)`` where (slope, intercept) is
    the deg-1 polynomial fit (numpy convention: highest order first) of
    *per-inference* delay vs N, suitable for
    :attr:`HardwareCoefficients.idlef_polynomial`.
    """
    import numpy as np
    import torch

    baseline_times = [_matmul_kernel() for _ in range(n_trials)]
    baseline_mean = float(np.median(baseline_times))
    raw_data: dict[int, float] = {1: baseline_mean}

    for n in range(2, n_max + 1):
        streams = [torch.cuda.Stream() for _ in range(n)]
        trial_times = []
        for _ in range(n_trials):
            with ThreadPoolExecutor(max_workers=n) as exe:
                futures = [exe.submit(_matmul_kernel, 512, 50, s) for s in streams]
                results = [f.result() for f in futures]
            trial_times.append(float(np.median(results)))
        raw_data[n] = float(np.median(trial_times))

    ns = np.array(sorted(k for k in raw_data.keys() if k >= 2), dtype=float)
    delays = np.array([raw_data[int(n)] - baseline_mean for n in ns])
    coefs = np.polyfit(ns, delays, deg=1)
    return (float(coefs[0]), float(coefs[1])), raw_data


def _sample_power_freq() -> tuple[float, float] | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=power.draw,clocks.current.graphics",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        parts = result.stdout.strip().split(",")
        if len(parts) >= 2:
            return float(parts[0].strip()), float(parts[1].strip())
    except Exception:
        return None
    return None


def _sustained_load(duration_s: float, size: int, stop_event: threading.Event) -> None:
    import torch

    device = torch.device("cuda")
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    start = time.perf_counter()
    while time.perf_counter() - start < duration_s and not stop_event.is_set():
        _ = a @ b
        torch.cuda.synchronize()


def _measure_powerp_linear(
    tdp_watts: float, sustain_seconds: float = 20.0, sample_interval_ms: int = 100
) -> tuple[float, list[tuple[float, float]]]:
    """Powerp sweep: GPU freq vs power.draw slope above TDP.

    Returns ``(slope_mhz_per_watt, raw_samples)`` where the slope is
    suitable for :attr:`HardwareCoefficients.powerp_linear`.
    """
    import numpy as np

    samples: list[tuple[float, float]] = []
    stop_event = threading.Event()

    def collect():
        while not stop_event.is_set():
            s = _sample_power_freq()
            if s is not None:
                samples.append(s)
            time.sleep(sample_interval_ms / 1000.0)

    collector = threading.Thread(target=collect, daemon=True)
    collector.start()
    load_thread = threading.Thread(target=_sustained_load, args=(sustain_seconds, 2048, stop_event))
    load_thread.start()
    load_thread.join()
    stop_event.set()
    collector.join(timeout=2)

    samples_above = [(p, f) for p, f in samples if p > tdp_watts]
    if len(samples_above) < 3:
        samples_above = samples
    if not samples_above:
        return 0.0, samples
    powers = np.array([p for p, _ in samples_above])
    freqs = np.array([f for _, f in samples_above])
    if len(powers) >= 2 and powers.std() > 1.0:
        a, _b = np.polyfit(powers, freqs, deg=1)
        slope = float(a)
    else:
        slope = 0.0
    return slope, samples


def sweep_hardware(
    *,
    output_dir: Path | None = None,
    n_idlef_max: int = 5,
    n_idlef_trials: int = 5,
    powerp_sustain_seconds: float = 20.0,
    substrate_name: str | None = None,
) -> HardwareCoefficients:
    """Fit :class:`HardwareCoefficients` for the current substrate.

    Parameters
    ----------
    output_dir
        Optional directory to dump raw sweep data
        (``hardware_idlef_raw.json`` + ``hardware_powerp_raw.json``).
    n_idlef_max
        Maximum N for the idlef concurrency sweep (default 5).
    n_idlef_trials
        Trials per N in the idlef sweep (default 5).
    powerp_sustain_seconds
        Seconds of sustained heavy load for the powerp sweep (default 20).
    substrate_name
        Optional override for the substrate name field. If ``None``,
        derived from ``nvidia-smi --query-gpu=name`` (lowercased with
        underscores).

    Returns
    -------
    HardwareCoefficients
        Frozen dataclass with operator's substrate constants.

    Raises
    ------
    ImportError
        If ``torch`` / ``numpy`` are not installed (extras
        ``[calibration]`` required plus a PyTorch matching the operator's
        CUDA).
    RuntimeError
        If CUDA is unavailable or ``nvidia-smi`` query fails.
    """
    _require_runtime()
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; nvinx calibration requires a usable GPU")

    smi = _query_nvidia_smi()
    name = substrate_name or smi["substrate_name"]
    nominal_freq_mhz = float(smi["nominal_freq_mhz"])
    tdp_watts = float(smi["tdp_watts"])

    for _ in range(3):
        _matmul_kernel()  # warm-up

    idlef_poly, idlef_raw = _measure_idlef_polynomial(n_max=n_idlef_max, n_trials=n_idlef_trials)
    powerp_slope, powerp_raw = _measure_powerp_linear(
        tdp_watts=tdp_watts, sustain_seconds=powerp_sustain_seconds
    )

    coefs = HardwareCoefficients(
        idlef_polynomial=idlef_poly,
        powerp_linear=(powerp_slope,),
        nominal_freq_mhz=nominal_freq_mhz,
        tdp_watts=tdp_watts,
        substrate_name=str(name),
    )

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "hardware_idlef_raw.json").write_text(
            json.dumps({"raw_data": idlef_raw, "polynomial": list(idlef_poly)}, indent=2)
        )
        (output_dir / "hardware_powerp_raw.json").write_text(
            json.dumps({"raw_samples": powerp_raw, "slope_mhz_per_watt": powerp_slope}, indent=2)
        )
        (output_dir / "hardware_constants.json").write_text(
            json.dumps(
                {
                    "idlef_polynomial": list(idlef_poly),
                    "powerp_linear": [powerp_slope],
                    "nominal_freq_mhz": nominal_freq_mhz,
                    "tdp_watts": tdp_watts,
                    "substrate_name": str(name),
                },
                indent=2,
            )
        )
    return coefs
