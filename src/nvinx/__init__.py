"""nvinx — nginx-style workload scheduler for limited-VRAM GPU bench setups."""

from nvinx.catalog import HardwareSpec, ModelSpec, Residency, SchedulingPlan
from nvinx.patterns import fractional_coresidency, ram_overflow, serial_handoff

__version__ = "0.1.0"

__all__ = [
    "HardwareSpec",
    "ModelSpec",
    "Residency",
    "SchedulingPlan",
    "fractional_coresidency",
    "ram_overflow",
    "serial_handoff",
    "__version__",
]
