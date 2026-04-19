"""A composed-day example: use all three patterns across one research day.

A realistic day on a 4 GB bench:

- Overnight window (Pattern A): a long inference holds the GPU; classical CPU
  pipelines run in parallel.
- Morning window (Pattern B): a RAG stack of small models co-resides in VRAM.
- Afternoon window (Pattern C): a large folding model exceeds VRAM, so layers
  spill to system RAM via `accelerate`.

nvinx gives placement decisions one at a time; your orchestrator drives the
transitions between windows.
"""

from nvinx.catalog import HardwareSpec, ModelSpec, Residency
from nvinx.patterns import fractional_coresidency, ram_overflow, serial_handoff

HW = HardwareSpec(vram_gb=4.0, ram_gb=32.0, cpu_cores=8)


def window_overnight():
    print("\n--- Window 1: overnight (Pattern A — serial_handoff) ---")
    gpu_job = ModelSpec(name="long_inference", vram_gb=3.5, residency=Residency.GPU_EXCLUSIVE)
    cpu_jobs = [
        ModelSpec(name="variant_caller", vram_gb=0.0, residency=Residency.CPU_ONLY),
        ModelSpec(name="qc_reporter", vram_gb=0.0, residency=Residency.CPU_ONLY),
    ]
    plan = serial_handoff(gpu_job, cpu_jobs, HW)
    print(f"  GPU held by    : {plan.gpu_foreground.name}")
    print(f"  CPU parallel   : {[m.name for m in plan.cpu_parallel]}")


def window_morning():
    print("\n--- Window 2: morning (Pattern B — fractional_coresidency) ---")
    candidates = [
        ModelSpec(name="classifier_a", vram_gb=0.7, residency=Residency.GPU_SHARED),
        ModelSpec(name="embedder", vram_gb=0.3, residency=Residency.GPU_SHARED),
        ModelSpec(name="reranker", vram_gb=1.2, residency=Residency.GPU_SHARED),
    ]
    plan = fractional_coresidency(candidates, HW)
    print(f"  GPU co-resident: {[m.name for m in plan.gpu_coresident]}")


def window_afternoon():
    print("\n--- Window 3: afternoon (Pattern C — ram_overflow) ---")
    big_model = ModelSpec(
        name="big_folder",
        vram_gb=6.0,
        residency=Residency.GPU_RAM_OVERFLOW,
        ram_overflow_supported=True,
        ram_gb_needed=10.0,
    )
    hint = ram_overflow(big_model, HW)
    print(f"  device_map     : {hint['device_map']}")
    print(f"  max_memory     : {hint['max_memory']}")


def main() -> None:
    print(f"Hardware: VRAM={HW.vram_gb} GB, RAM={HW.ram_gb} GB, CPU={HW.cpu_cores}c")
    window_overnight()
    window_morning()
    window_afternoon()


if __name__ == "__main__":
    main()
