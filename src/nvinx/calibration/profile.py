"""Per-model standalone profiling.

Profiles a single model in isolation to produce an
:class:`nvinx.interference.InterferenceProfile` with ``kernels``,
``baseidle_ms``, ``act_solo_ms``, ``l2_saturation_pct``, and ``power_w``
fields populated. ``theta`` is left as ``None`` — fit it later via
:func:`nvinx.calibration.fit.fit_thetas` once you have cross-pair
measurements.

Operator API
------------

The operator supplies a **loader callable** that returns a
:class:`ProfileTarget` bundling: a unique name, an ``inference_fn`` that
runs one inference, a ``sample_input`` already on GPU, and a
``loader_code`` Python source string the ncu subprocess can execute to
re-build the same ``inference_fn`` + ``sample_input`` in isolation.

The factory pattern — calling the loader inside ``profile_model`` rather
than passing pre-loaded artifacts in — lets VRAM-tight benches free the
parent process's model copy before the ncu subprocess spawns its own copy
of the same model.
"""

from __future__ import annotations

import gc
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nvinx.interference import InterferenceProfile


@dataclass
class ProfileTarget:
    """Bundle of artifacts needed to profile a single model.

    Parameters
    ----------
    name
        Identifier (used as :attr:`InterferenceProfile.name` and as a key
        in pair-measurement lookups).
    inference_fn
        Zero-shot inference callable. Takes ``sample_input``; return value
        is ignored. Will be called multiple times for warm-up + measure.
    sample_input
        Pre-loaded sample input on GPU. Will be reused across all
        inferences (warm-up + measure).
    loader_code
        Self-contained Python source string that defines ``inference_fn``
        and ``sample_input`` in module scope. Executed inside the ncu
        subprocess; must include any framework imports.
    canonical_batch
        Batch size encoded in ``sample_input`` (default 1). Advisory; the
        profile is bound to this batch size.
    """

    name: str
    inference_fn: Callable[[Any], Any]
    sample_input: Any
    loader_code: str = ""
    canonical_batch: int = 1


def _require_runtime():
    try:
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "nvinx.calibration.profile requires numpy + torch. "
            "Install scipy + numpy via: pip install nvinx[calibration] "
            "and install torch matching your CUDA via: "
            "https://pytorch.org/get-started/locally/"
        ) from e


def _measure_active_and_query(
    target: ProfileTarget, n_warmup: int, n_measure: int
) -> tuple[float, float]:
    """Return ``(active_time_ms_median, query_latency_ms_median)``."""
    import numpy as np
    import torch

    for _ in range(n_warmup):
        target.inference_fn(target.sample_input)
    torch.cuda.synchronize()
    active_times: list[float] = []
    query_latencies: list[float] = []
    for _ in range(n_measure):
        wall_start = time.perf_counter()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        target.inference_fn(target.sample_input)
        end.record()
        torch.cuda.synchronize()
        wall_end = time.perf_counter()
        active_times.append(start.elapsed_time(end))
        query_latencies.append((wall_end - wall_start) * 1000.0)
    return float(np.median(active_times)), float(np.median(query_latencies))


def _count_kernels(target: ProfileTarget) -> int:
    import torch

    target.inference_fn(target.sample_input)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=False,
    ) as prof:
        target.inference_fn(target.sample_input)
        torch.cuda.synchronize()
    events = prof.events()
    cuda_kernels = sum(
        1
        for e in events
        if getattr(e, "device_type", None) == torch.profiler.DeviceType.CUDA
        and getattr(e, "cuda_time", 0) > 0
    )
    if cuda_kernels == 0:
        cuda_kernels = sum(1 for e in events if getattr(e, "cuda_time", 0) > 0)
    return max(cuda_kernels, 1)


def _measure_power_w(target: ProfileTarget, n_seconds: float) -> float:
    import numpy as np
    import torch

    stop_event = threading.Event()
    samples: list[float] = []

    def collect():
        while not stop_event.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                samples.append(float(result.stdout.strip()))
            except Exception:
                pass
            time.sleep(0.1)

    collector = threading.Thread(target=collect, daemon=True)
    collector.start()
    start = time.perf_counter()
    while time.perf_counter() - start < n_seconds:
        target.inference_fn(target.sample_input)
        torch.cuda.synchronize()
    stop_event.set()
    collector.join(timeout=1)
    if not samples:
        return 0.0
    return float(np.median(samples))


def profile_model(
    loader: Callable[[], ProfileTarget],
    *,
    architecture_class: str = "unknown",
    n_warmup: int = 3,
    n_measure: int = 5,
    power_sustain_seconds: float = 5.0,
    ncu_sudo: bool = True,
    output_dir: Path | None = None,
) -> InterferenceProfile:
    """Profile one model standalone and return an :class:`InterferenceProfile`.

    Parameters
    ----------
    loader
        Zero-arg callable returning a fresh :class:`ProfileTarget`. Called
        twice (once for steps 1-3 + 5-6; once for the ncu subprocess in
        step 4 via its ``loader_code`` string). The factory pattern lets
        VRAM-tight benches free intermediate state.
    architecture_class
        Optional tag like ``"encoder_transformer"``. Advisory only; stored
        on the returned profile.
    n_warmup, n_measure
        Warm-up + measurement inference counts (median over n_measure
        trials feeds ``act_solo_ms``).
    power_sustain_seconds
        Sustained-load duration for the power measurement (default 5s).
    ncu_sudo
        Whether to invoke ncu via ``sudo -n`` (default True). Disable if
        ``NVreg_RestrictProfilingToAdminUsers=0`` is set on the driver.
    output_dir
        Optional directory to dump per-model JSON.

    Returns
    -------
    InterferenceProfile
        Frozen dataclass with ``theta=None`` — fit it later via
        :func:`nvinx.calibration.fit.fit_thetas`.

    Raises
    ------
    ImportError
        If torch / numpy are not installed.
    RuntimeError
        If CUDA is unavailable.
    ValueError
        If the target's ``loader_code`` field is empty (required for ncu
        subprocess in step 4).
    """
    _require_runtime()
    import torch

    from nvinx.calibration.ncu import measure_l2_saturation

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available; nvinx calibration requires a usable GPU")

    target = loader()
    if not target.loader_code:
        raise ValueError(
            f"target {target.name!r} has empty loader_code; required for ncu subprocess"
        )

    act_ms, query_ms = _measure_active_and_query(target, n_warmup, n_measure)
    baseidle_ms = max(query_ms - act_ms, 0.0)
    n_kernels = _count_kernels(target)
    power_w = _measure_power_w(target, n_seconds=power_sustain_seconds)

    # VRAM-tight escape: free parent's model before ncu subprocess
    target_name = target.name
    loader_code = target.loader_code
    canonical_batch = target.canonical_batch
    target.inference_fn = lambda x: None  # type: ignore[assignment]
    target.sample_input = None  # type: ignore[assignment]
    del target
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    l2_pct, _n_kernels_ncu = measure_l2_saturation(
        loader_code, n_inferences=2, sudo=ncu_sudo, output_dir=output_dir
    )

    profile = InterferenceProfile(
        name=target_name,
        kernels=n_kernels,
        baseidle_ms=baseidle_ms,
        act_solo_ms=act_ms,
        l2_saturation_pct=l2_pct,
        power_w=power_w,
        architecture_class=architecture_class,
        theta=None,
    )

    if output_dir is not None:
        import json as _json

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / f"model_{target_name}.json").write_text(
            _json.dumps(
                {
                    "name": target_name,
                    "kernels": n_kernels,
                    "baseidle_ms": baseidle_ms,
                    "act_solo_ms": act_ms,
                    "l2_saturation_pct": l2_pct,
                    "power_w": power_w,
                    "architecture_class": architecture_class,
                    "canonical_batch": canonical_batch,
                },
                indent=2,
            )
        )
    return profile
