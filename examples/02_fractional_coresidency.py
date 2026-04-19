"""Pattern B — fractional_coresidency: pack small-VRAM models into a shared budget.

Scenario: a RAG pipeline needs a classifier + an embedder + a reranker. Each fits
in VRAM alongside the others. Instead of load/unload churn, pack them into one
co-resident window.
"""

from nvinx.catalog import HardwareSpec, ModelSpec, Residency
from nvinx.patterns import fractional_coresidency


def main() -> None:
    hw = HardwareSpec(vram_gb=4.0, ram_gb=32.0, cpu_cores=8)

    candidates = [
        ModelSpec(name="classifier_a", vram_gb=0.7, residency=Residency.GPU_SHARED),
        ModelSpec(name="embedder", vram_gb=0.3, residency=Residency.GPU_SHARED),
        ModelSpec(name="reranker", vram_gb=1.2, residency=Residency.GPU_SHARED),
        ModelSpec(
            name="oversized_model",
            vram_gb=4.0,
            residency=Residency.GPU_SHARED,
            cpu_fallback_supported=True,
        ),
    ]

    plan = fractional_coresidency(candidates, hw)

    print(f"GPU co-resident: {[m.name for m in plan.gpu_coresident]}")
    print(f"CPU parallel   : {[m.name for m in plan.cpu_parallel]}")
    print(f"Unscheduled    : {[m.name for m in plan.unscheduled]}")
    for note in plan.notes:
        print(f"Note           : {note}")


if __name__ == "__main__":
    main()
