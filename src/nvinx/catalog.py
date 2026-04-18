"""Data model for hardware and model specifications consumed by patterns."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Residency(Enum):
    """How a model can occupy compute resources."""

    GPU_EXCLUSIVE = "gpu_exclusive"
    GPU_SHARED = "gpu_shared"
    CPU_ONLY = "cpu_only"
    GPU_RAM_OVERFLOW = "gpu_ram_overflow"


@dataclass
class ModelSpec:
    """A workload to schedule."""

    name: str
    vram_gb: float
    residency: Residency
    cpu_fallback_supported: bool = False
    ram_overflow_supported: bool = False
    ram_gb_needed: float | None = None


@dataclass
class HardwareSpec:
    """The physical bench envelope."""

    vram_gb: float
    ram_gb: float
    cpu_cores: int


@dataclass
class SchedulingPlan:
    """Output of a pattern call — a declarative placement for a runtime to execute."""

    gpu_foreground: ModelSpec | None = None
    gpu_coresident: list[ModelSpec] = field(default_factory=list)
    cpu_parallel: list[ModelSpec] = field(default_factory=list)
    overflow: list[ModelSpec] = field(default_factory=list)
    unscheduled: list[ModelSpec] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
