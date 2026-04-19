"""Smoke tests for the three scheduling patterns."""

import pytest

from nvinx.catalog import HardwareSpec, ModelSpec, Residency
from nvinx.patterns import fractional_coresidency, ram_overflow, serial_handoff

HW_4GB = HardwareSpec(vram_gb=4.0, ram_gb=32.0, cpu_cores=8)


def test_serial_handoff_places_cpu_candidates_in_parallel():
    gpu_job = ModelSpec(name="basecaller", vram_gb=3.0, residency=Residency.GPU_EXCLUSIVE)
    cpu_jobs = [
        ModelSpec(name="variant_caller", vram_gb=0.0, residency=Residency.CPU_ONLY),
        ModelSpec(name="report_writer", vram_gb=0.0, residency=Residency.CPU_ONLY),
    ]
    plan = serial_handoff(gpu_job, cpu_jobs, HW_4GB)

    assert plan.gpu_foreground is gpu_job
    assert len(plan.cpu_parallel) == 2
    assert plan.unscheduled == []


def test_serial_handoff_rejects_non_exclusive_gpu_job():
    shared = ModelSpec(name="embed", vram_gb=0.2, residency=Residency.GPU_SHARED)
    with pytest.raises(ValueError, match="GPU_EXCLUSIVE"):
        serial_handoff(shared, [], HW_4GB)


def test_serial_handoff_rejects_oversized_gpu_job():
    oversized = ModelSpec(name="giant", vram_gb=8.0, residency=Residency.GPU_EXCLUSIVE)
    with pytest.raises(ValueError, match="VRAM"):
        serial_handoff(oversized, [], HW_4GB)


def test_fractional_coresidency_packs_within_budget():
    candidates = [
        ModelSpec(name="small_a", vram_gb=0.5, residency=Residency.GPU_SHARED),
        ModelSpec(name="small_b", vram_gb=0.2, residency=Residency.GPU_SHARED),
        ModelSpec(
            name="big",
            vram_gb=4.0,
            residency=Residency.GPU_SHARED,
            cpu_fallback_supported=True,
        ),
    ]
    plan = fractional_coresidency(candidates, HW_4GB)

    coresident_names = {m.name for m in plan.gpu_coresident}
    assert coresident_names == {"small_a", "small_b"}
    assert len(plan.cpu_parallel) == 1
    assert plan.cpu_parallel[0].name == "big"


def test_fractional_coresidency_unscheduled_when_no_fallback():
    candidates = [
        ModelSpec(name="big_no_fallback", vram_gb=5.0, residency=Residency.GPU_SHARED),
    ]
    plan = fractional_coresidency(candidates, HW_4GB)

    assert plan.gpu_coresident == []
    assert len(plan.unscheduled) == 1


def test_ram_overflow_returns_accelerate_hint():
    model = ModelSpec(
        name="big_folder",
        vram_gb=6.0,
        residency=Residency.GPU_RAM_OVERFLOW,
        ram_overflow_supported=True,
        ram_gb_needed=10.0,
    )
    hint = ram_overflow(model, HW_4GB)

    assert hint["device_map"] == "auto"
    max_memory = hint["max_memory"]
    assert 0 in max_memory
    assert "cpu" in max_memory


def test_ram_overflow_rejects_unsupported_model():
    model = ModelSpec(
        name="not_offloadable",
        vram_gb=6.0,
        residency=Residency.GPU_RAM_OVERFLOW,
        ram_overflow_supported=False,
    )
    with pytest.raises(ValueError, match="ram_overflow_supported"):
        ram_overflow(model, HW_4GB)


def test_ram_overflow_rejects_oversized_ram_requirement():
    model = ModelSpec(
        name="needs_too_much_ram",
        vram_gb=6.0,
        residency=Residency.GPU_RAM_OVERFLOW,
        ram_overflow_supported=True,
        ram_gb_needed=64.0,
    )
    with pytest.raises(ValueError, match="RAM"):
        ram_overflow(model, HW_4GB)
