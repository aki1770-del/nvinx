"""nvinx — nginx-style workload scheduler for limited-VRAM GPU bench setups."""

from nvinx.catalog import HardwareSpec, ModelSpec, Residency, SchedulingPlan
from nvinx.interference import (
    HardwareCoefficients,
    InterferenceProfile,
    PairLookupEntry,
    asymmetry_predictor,
    fit_gamma_kernel_size,
    lookup_pair_latency,
    max_kernel_rate_score,
    predict_pair_latency,
    predict_pair_latency_queue_aware,
    predict_pair_latency_queue_aware_v5,
)
from nvinx.patterns import (
    fractional_coresidency,
    fractional_coresidency_v2,
    ram_overflow,
    serial_handoff,
)

__version__ = "0.3.0a1"

__all__ = [
    # v0.1 catalog
    "HardwareSpec",
    "ModelSpec",
    "Residency",
    "SchedulingPlan",
    # v0.1 patterns
    "fractional_coresidency",
    "ram_overflow",
    "serial_handoff",
    # v0.2 interference primitives
    "HardwareCoefficients",
    "InterferenceProfile",
    "PairLookupEntry",
    "asymmetry_predictor",
    "lookup_pair_latency",
    "max_kernel_rate_score",
    "predict_pair_latency",
    "predict_pair_latency_queue_aware",
    # v0.6 alpha — substrate-bound V5 kernel-size-ratio correction (γ operator-fitted)
    "fit_gamma_kernel_size",
    "predict_pair_latency_queue_aware_v5",
    # v0.2 enhanced patterns
    "fractional_coresidency_v2",
    "__version__",
]
