"""Cross-pair co-located latency measurement.

Runs two pre-loaded :class:`ProfileTarget` objects concurrently on separate
CUDA streams; measures each side's wall-clock GPU active time under
contention; returns a tuple consumable by
:func:`nvinx.calibration.fit.fit_thetas` and :func:`fit_v5`.

Both targets must already be loaded on GPU — that's the operator's
constraint. The runner (:func:`nvinx.calibration.run_calibration`) handles
loader → target → validate sequencing.
"""

from __future__ import annotations

import threading

from nvinx.calibration.profile import ProfileTarget


def _require_runtime():
    try:
        import numpy as np  # noqa: F401
        import torch  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "nvinx.calibration.validate requires numpy + torch. "
            "Install via: pip install nvinx[calibration] "
            "(and a torch matching your CUDA)."
        ) from e


def validate_pair(
    target_a: ProfileTarget,
    target_b: ProfileTarget,
    *,
    n_warmup: int = 2,
    n_measure: int = 5,
) -> tuple[str, str, float, float]:
    """Co-locate ``target_a`` + ``target_b``; return per-side median latency.

    Runs both targets concurrently on separate CUDA streams (one stream per
    target, one Python thread per stream); measures GPU active time per
    side via CUDA events; returns median over ``n_measure`` trials.

    Parameters
    ----------
    target_a, target_b
        Pre-loaded :class:`ProfileTarget` objects. The caller must have
        invoked the respective loaders and is responsible for their VRAM
        lifecycle.
    n_warmup
        Concurrent warm-up trials before measurement begins (default 2).
    n_measure
        Measured concurrent trials. Median feeds the returned tuple.

    Returns
    -------
    tuple
        ``(target_a.name, target_b.name, meas_a_ms, meas_b_ms)`` — directly
        appendable to the ``pair_measurements`` list consumed by
        :func:`fit_thetas` / :func:`fit_v5` / :func:`lopo_cross_validate`.

    Raises
    ------
    ImportError
        If torch / numpy are not installed.
    """
    _require_runtime()
    import numpy as np
    import torch

    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()

    for _ in range(n_warmup):
        with torch.cuda.stream(stream_a):
            target_a.inference_fn(target_a.sample_input)
        with torch.cuda.stream(stream_b):
            target_b.inference_fn(target_b.sample_input)
        torch.cuda.synchronize()

    times_a: list[float] = []
    times_b: list[float] = []

    for _ in range(n_measure):
        ev_a_start = torch.cuda.Event(enable_timing=True)
        ev_a_end = torch.cuda.Event(enable_timing=True)
        ev_b_start = torch.cuda.Event(enable_timing=True)
        ev_b_end = torch.cuda.Event(enable_timing=True)

        def run_a(start=ev_a_start, end=ev_a_end):
            with torch.cuda.stream(stream_a):
                start.record(stream_a)
                target_a.inference_fn(target_a.sample_input)
                end.record(stream_a)

        def run_b(start=ev_b_start, end=ev_b_end):
            with torch.cuda.stream(stream_b):
                start.record(stream_b)
                target_b.inference_fn(target_b.sample_input)
                end.record(stream_b)

        thread_a = threading.Thread(target=run_a)
        thread_b = threading.Thread(target=run_b)
        thread_a.start()
        thread_b.start()
        thread_a.join()
        thread_b.join()
        stream_a.synchronize()
        stream_b.synchronize()
        times_a.append(ev_a_start.elapsed_time(ev_a_end))
        times_b.append(ev_b_start.elapsed_time(ev_b_end))

    return (
        target_a.name,
        target_b.name,
        float(np.median(times_a)),
        float(np.median(times_b)),
    )
