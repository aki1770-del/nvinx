"""Nsight Compute (``ncu``) wrapper for L2 saturation measurement.

Measures ``lts__t_sectors.avg.pct_of_peak_sustained_elapsed`` weighted by
``gpu__time_duration.sum`` across all CUDA kernels in a model's inference
pass. The weighted average ("L2 saturation pct") feeds the
``InterferenceProfile.l2_saturation_pct`` field.

System prerequisite
-------------------

``ncu`` (Nsight Compute) installed and accessible::

    sudo apt install nsight-compute        # Ubuntu (multiverse repo)

The NVIDIA driver restricts performance counters to root by default
(``ERR_NVGPUCTRPERM``). Two ways to grant access:

1. Run ncu via ``sudo`` (set ``sudo=True`` in :func:`measure_l2_saturation`),
   or pre-arrange passwordless ``sudo ncu`` for the calibration user.
2. Load the NVIDIA kernel module with
   ``NVreg_RestrictProfilingToAdminUsers=0`` (typically via ``/etc/modprobe.d``).

The CSV parser is permissive about non-zero ncu exit codes — cuDNN cleanup
warnings at script exit are common and don't invalidate the measurement
data already collected. Real ncu errors manifest as missing CSV header.
"""

from __future__ import annotations

import csv
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def make_ncu_target_script(loader_code: str, n_inferences: int = 3) -> str:
    """Generate a self-contained Python script that ncu can profile.

    ``loader_code`` must define ``inference_fn`` and ``sample_input``. The
    generated script runs 2 warm-up inferences (excluded from ncu's profile
    by virtue of being identical work patterns; ncu samples every kernel
    regardless), then ``n_inferences`` measured inferences.
    """
    return f"""
import torch
{loader_code}

for _ in range(2):
    inference_fn(sample_input)
torch.cuda.synchronize()

for _ in range({n_inferences}):
    inference_fn(sample_input)
    torch.cuda.synchronize()
"""


def run_ncu_and_parse(
    script_text: str,
    *,
    venv_python: str | None = None,
    metrics: list[str] | None = None,
    sudo: bool = True,
    timeout_s: int = 900,
) -> tuple[float, int, list[dict]]:
    """Run ncu on ``script_text``; parse per-kernel L2 saturation + duration.

    Returns
    -------
    tuple
        ``(l2_saturation_pct_weighted_avg, n_kernels, raw_records)``.

        ``l2_saturation_pct_weighted_avg = sum(l2_sat_pct[k] * duration[k]) /
        sum(duration[k])`` over all kernels in the inference.

    Raises
    ------
    RuntimeError
        If ncu produces no CSV-parseable output.
    """
    if metrics is None:
        metrics = [
            "lts__t_sectors.avg.pct_of_peak_sustained_elapsed",
            "gpu__time_duration.sum",
        ]
    if venv_python is None:
        venv_python = sys.executable

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script_text)
        script_path = f.name

    try:
        cmd: list[str] = []
        if sudo:
            cmd += [
                "sudo",
                "-n",
                "env",
                f"PATH={os.environ.get('PATH', '')}",
                f"HOME={os.environ.get('HOME', '')}",
            ]
        cmd += [
            "ncu",
            "--csv",
            "--metrics",
            ",".join(metrics),
            "--target-processes",
            "all",
            venv_python,
            script_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        output = result.stdout
    finally:
        Path(script_path).unlink(missing_ok=True)

    lines = output.splitlines()
    csv_start: int | None = None
    for i, line in enumerate(lines):
        if line.startswith('"ID"'):
            csv_start = i
            break
    if csv_start is None:
        raise RuntimeError(
            f"ncu produced no CSV header (returncode={result.returncode}); "
            f"stderr tail: {result.stderr[-500:]}"
        )

    csv_text = "\n".join(lines[csv_start:])
    reader = csv.DictReader(io.StringIO(csv_text))

    kernels: dict[int, dict] = {}
    for row in reader:
        try:
            kid = int(row["ID"])
        except (ValueError, KeyError):
            continue
        metric = row.get("Metric Name", "")
        try:
            value = float(row.get("Metric Value", "0"))
        except ValueError:
            value = 0.0
        if kid not in kernels:
            kernels[kid] = {"name": row.get("Kernel Name", "")}
        if metric == "gpu__time_duration.sum":
            kernels[kid]["duration_ns"] = value
        elif metric == "lts__t_sectors.avg.pct_of_peak_sustained_elapsed":
            kernels[kid]["l2_sat_pct"] = value

    if not kernels:
        raise RuntimeError("ncu output had no parseable kernel rows")

    total_dur = 0.0
    weighted_sum = 0.0
    raw_records: list[dict] = []
    for kid, k in sorted(kernels.items()):
        dur = k.get("duration_ns", 0.0)
        sat = k.get("l2_sat_pct", 0.0)
        total_dur += dur
        weighted_sum += sat * dur
        raw_records.append(
            {"kernel_id": kid, "name": k.get("name", ""), "duration_ns": dur, "l2_sat_pct": sat}
        )

    if total_dur == 0.0:
        return 0.0, len(kernels), raw_records
    return weighted_sum / total_dur, len(kernels), raw_records


def measure_l2_saturation(
    loader_code: str,
    *,
    venv_python: str | None = None,
    n_inferences: int = 3,
    sudo: bool = True,
    output_dir: Path | None = None,
) -> tuple[float, int]:
    """Convenience wrapper: build ncu target script + run + return summary.

    Parameters
    ----------
    loader_code
        Self-contained Python source (string) that defines ``inference_fn``
        and ``sample_input`` in module scope. The ncu subprocess executes
        it, then runs the inference loop. Must include any framework
        imports.
    venv_python
        Path to the Python interpreter the ncu subprocess should use.
        Defaults to ``sys.executable`` (whichever interpreter is running
        the calibration).
    n_inferences
        Number of measured inferences (default 3). ncu profiles every
        kernel; more inferences yield more reproducible numbers but take
        longer (each model inference takes minutes under ncu).
    sudo
        Whether to invoke ncu via ``sudo -n`` (passwordless). Set to
        ``False`` if ``NVreg_RestrictProfilingToAdminUsers=0`` is set on
        the driver.
    output_dir
        Optional path to dump the raw ncu CSV for the operator's records.

    Returns
    -------
    tuple
        ``(l2_saturation_pct_weighted_avg, n_kernels)``.
    """
    script = make_ncu_target_script(loader_code, n_inferences)
    l2_pct, n_kernels, raw_records = run_ncu_and_parse(script, venv_python=venv_python, sudo=sudo)
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        import json as _json

        (output_dir / "ncu_l2_kernels.json").write_text(
            _json.dumps(
                {
                    "l2_saturation_pct_weighted_avg": l2_pct,
                    "n_kernels": n_kernels,
                    "kernels": raw_records,
                },
                indent=2,
            )
        )
    return l2_pct, n_kernels
