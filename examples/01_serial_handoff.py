"""Pattern A — serial_handoff: CPU work during a GPU-exclusive window.

Scenario: a 6-hour GPU-bound job (e.g. a long inference or basecaller run) holds
the GPU. Several CPU-only workloads can execute in parallel in the same wall-clock
window instead of waiting their turn.
"""

from nvinx.catalog import HardwareSpec, ModelSpec, Residency
from nvinx.patterns import serial_handoff


def main() -> None:
    hw = HardwareSpec(vram_gb=4.0, ram_gb=32.0, cpu_cores=8)

    gpu_job = ModelSpec(
        name="long_inference",
        vram_gb=3.5,
        residency=Residency.GPU_EXCLUSIVE,
    )

    cpu_candidates = [
        ModelSpec(name="feature_extractor", vram_gb=0.0, residency=Residency.CPU_ONLY),
        ModelSpec(name="report_writer", vram_gb=0.0, residency=Residency.CPU_ONLY),
        ModelSpec(name="vector_index_build", vram_gb=0.0, residency=Residency.CPU_ONLY),
    ]

    plan = serial_handoff(gpu_job, cpu_candidates, hw)

    print(f"GPU foreground: {plan.gpu_foreground.name}")
    print(f"CPU parallel  : {[m.name for m in plan.cpu_parallel]}")
    print(f"Unscheduled   : {[m.name for m in plan.unscheduled]}")
    for note in plan.notes:
        print(f"Note          : {note}")


if __name__ == "__main__":
    main()
