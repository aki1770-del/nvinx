"""Substrate-class detection for `nvinx` v0.3.0a2 substrate-class gate.

The v0.5 / v0.6 calibration corpus on RTX A1000 4 GB mobile (16 SMs, 4 MB L2
cache, Ampere sm_86) produced γ_kernel_size ≈ 0.7456 for the V5 kernel-size-
ratio correction. A first-party cross-substrate validation on NVIDIA A100
SXM4 40 GB (108 SMs, 40 MB L2 cache, compute capability 8.0) in 2026-05-15
measured γ_kernel_size = 0.0331 — a 22.5× collapse. V5 LOPO was WORSE than
V1 LOPO under the A100 measurement (15.54 % vs 14.67 %). The conclusion:
V5 is mobile-specific.

This module provides best-effort runtime detection of the substrate class
so operators on datacenter-class GPUs are warned against using the published
mobile-substrate reference γ. Detection is via NVML (compute_capability,
SM count, total memory). Mobile-class is sm_86 (Ampere mobile / RTX 30
mobile + A1000 mobile) and similar small-VRAM small-SM substrates. Anything
sm_70/sm_80/sm_90+ with SM count > 64 is datacenter-class.

The gate is ADVISORY: it does NOT silently override an explicit
``gamma_kernel_size`` argument. Per the operator-controlled discipline
documented in `docs/calibrating-your-substrate.md`, the operator's
``gamma_kernel_size`` value is always respected. The gate emits a
``RuntimeWarning`` when a non-zero γ is passed on a detected
datacenter-class substrate, alerting the operator to the substrate-class
mismatch.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

SubstrateClass = Literal["datacenter", "mobile", "unknown"]


@dataclass(frozen=True)
class SubstrateInfo:
    """Detected substrate parameters."""

    name: str
    sm_count: int | None
    total_memory_mb: int | None
    compute_capability_major: int | None
    compute_capability_minor: int | None
    substrate_class: SubstrateClass

    def is_datacenter(self) -> bool:
        return self.substrate_class == "datacenter"

    def is_mobile(self) -> bool:
        return self.substrate_class == "mobile"


# Thresholds for substrate-class detection.
# Datacenter: SM count > 64 (A100 has 108, V100 has 80, H100 has 132,
# A6000 has 84, A10 has 72) AND total memory >= 16 GB
# Mobile: SM count <= 48 (A1000 has 16, RTX 3070 mobile has 40,
# RTX 3080 mobile has 48) OR total memory < 12 GB
_DATACENTER_SM_THRESHOLD = 64
_DATACENTER_MEMORY_MB_THRESHOLD = 16 * 1024
_MOBILE_SM_THRESHOLD = 48
_MOBILE_MEMORY_MB_THRESHOLD = 12 * 1024


def _classify(
    sm_count: int | None,
    total_memory_mb: int | None,
) -> SubstrateClass:
    """Apply the SM-count + memory thresholds to classify the substrate.

    Returns "datacenter", "mobile", or "unknown".
    """
    if sm_count is None and total_memory_mb is None:
        return "unknown"

    # Datacenter: explicitly many-SM + much memory
    if (
        sm_count is not None
        and sm_count > _DATACENTER_SM_THRESHOLD
        and total_memory_mb is not None
        and total_memory_mb >= _DATACENTER_MEMORY_MB_THRESHOLD
    ):
        return "datacenter"

    # Mobile: explicitly few-SM OR small memory
    if (
        sm_count is not None
        and sm_count <= _MOBILE_SM_THRESHOLD
        or total_memory_mb is not None
        and total_memory_mb < _MOBILE_MEMORY_MB_THRESHOLD
    ):
        return "mobile"

    # In-between (e.g., 56-SM RTX 4080 mobile with 12 GB; RTX 4090 mobile with 16 GB)
    # — do not warn; operator decides.
    return "unknown"


@lru_cache(maxsize=1)
def detect_substrate(device_index: int = 0) -> SubstrateInfo:
    """Detect the substrate class for the GPU at ``device_index`` via NVML.

    Cached at module-level; called once per process.

    Returns
    -------
    SubstrateInfo
        Best-effort detection. If NVML or pynvml is unavailable, returns
        ``SubstrateInfo(name="(unknown)", sm_count=None, total_memory_mb=None,
        compute_capability_major=None, compute_capability_minor=None,
        substrate_class="unknown")``.
    """
    try:
        import pynvml  # type: ignore[import-not-found]
    except ImportError:
        return SubstrateInfo(
            name="(pynvml not installed)",
            sm_count=None,
            total_memory_mb=None,
            compute_capability_major=None,
            compute_capability_minor=None,
            substrate_class="unknown",
        )

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        name_raw = pynvml.nvmlDeviceGetName(handle)
        name = name_raw.decode() if isinstance(name_raw, bytes) else name_raw
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_mb = int(mem_info.total / (1024 * 1024))

        # Try compute capability first (more robust)
        cc_major: int | None = None
        cc_minor: int | None = None
        try:
            cc_major, cc_minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        except Exception:
            pass

        # Try SM count
        sm_count: int | None = None
        try:
            # newer pynvml exposes this as a constant query
            sm_count = pynvml.nvmlDeviceGetNumGpuCores(handle)
            # nvmlDeviceGetNumGpuCores returns CUDA cores, not SMs;
            # divide by per-SM cores (128 for Ampere/Hopper; 64 for Volta)
            if cc_major == 7:  # Volta
                sm_count = sm_count // 64
            elif cc_major and cc_major >= 8:  # Ampere/Ada/Hopper
                sm_count = sm_count // 128
            else:
                sm_count = None  # uncertain
        except Exception:
            sm_count = None

    except Exception as e:
        return SubstrateInfo(
            name=f"(nvml error: {type(e).__name__})",
            sm_count=None,
            total_memory_mb=None,
            compute_capability_major=None,
            compute_capability_minor=None,
            substrate_class="unknown",
        )

    substrate_class = _classify(sm_count, total_mb)

    return SubstrateInfo(
        name=name,
        sm_count=sm_count,
        total_memory_mb=total_mb,
        compute_capability_major=cc_major,
        compute_capability_minor=cc_minor,
        substrate_class=substrate_class,
    )


def warn_if_datacenter_with_nonzero_gamma(
    gamma_kernel_size: float | None,
    *,
    device_index: int = 0,
) -> None:
    """Emit a RuntimeWarning if ``gamma_kernel_size`` is non-zero AND the
    detected substrate is datacenter-class.

    Per Track C 2026-05-15 first-party measurement: γ ≈ 0.75 (A1000 mobile)
    collapsed to γ ≈ 0.03 on A100 SXM4 40 GB. Using a non-zero ``gamma_kernel_size``
    from a mobile-substrate fit on a datacenter substrate WILL DEGRADE
    queue-aware predictions on that substrate (V5 LOPO worse than V1 LOPO).

    The warning does NOT override the operator's value. Per
    ``docs/calibrating-your-substrate.md`` operator-controlled discipline,
    the operator's ``gamma_kernel_size`` is always respected. The warning
    alerts the operator that they may want to refit on the actual substrate
    (or pass 0/None on this substrate).

    Suppress via ``warnings.filterwarnings("ignore", category=RuntimeWarning,
    module="nvinx.substrate")``.
    """
    if gamma_kernel_size is None or gamma_kernel_size == 0.0:
        return

    info = detect_substrate(device_index=device_index)
    if info.is_datacenter():
        warnings.warn(
            (
                f"nvinx v0.3.0a2 substrate-class gate: detected datacenter-class "
                f"substrate ({info.name}; {info.sm_count} SMs; "
                f"{info.total_memory_mb} MB VRAM; CC {info.compute_capability_major}."
                f"{info.compute_capability_minor}) but gamma_kernel_size="
                f"{gamma_kernel_size!r} (non-zero) was passed. The v0.5 / v0.6 "
                f"reference γ ≈ 0.75 was fitted on a 4 GB RTX A1000 mobile + "
                f"transformer/LLM corpus and EMPIRICALLY DOES NOT TRANSFER to "
                f"datacenter substrates (Track C 2026-05-15: γ collapsed to "
                f"~0.033 on A100 SXM4 40 GB; V5 LOPO worse than V1 LOPO). "
                f"Consider passing gamma_kernel_size=0 (reduces to V1 queue-aware "
                f"baseline) or fitting your own γ on this substrate via "
                f"nvinx.calibration. Operator override respected."
            ),
            category=RuntimeWarning,
            stacklevel=3,
        )
