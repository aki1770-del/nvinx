"""Case study — small oncology research bench, 4 GB VRAM workstation.

A worked example of nvinx scheduling the kind of multi-model heterogeneous
workload that runs on a single 4 GB GPU oncology research bench. The workload
has four distinct windows across a research day; each maps onto one of the
three nvinx patterns.

This case study illustrates the nvinx target population: research benches
running NON-LLM, NON-image-generation workloads — bioinformatics inference,
neoantigen ranking, protein structure prediction — that are not served by
LLM-coupled runtimes (exllamav3-class) or image-generation-coupled runtimes
(ComfyUI-MultiGPU-class). The patterns are framework-agnostic; the bench
runs PyTorch + HuggingFace + a few specialized inference tools, each
independently chosen by the researcher.

Workload composition (representative; not tied to a specific institution or
dataset):

  Window 1 (overnight) — long-read sequencing basecaller (GPU-EXCLUSIVE).
                          During the 4-8 hour window, CPU-only post-processing
                          runs in parallel (variant calling, QC reporting,
                          document scaffolding).

  Window 2 (mid-day)   — neoantigen ranking ensemble: a small
                          mutation-classifier model (~0.5 GB VRAM) plus an
                          MHC-I binding predictor (~0.2 GB VRAM). Both fit
                          alongside each other in the 4 GB envelope.

  Window 3 (mid-day)   — structure prediction for the top-3 candidates from
                          Window 2 (~2.2 GB VRAM, GPU-EXCLUSIVE for the
                          residue-bound case).

  Window 4 (occasional) — a candidate exceeds 216 residues; structure
                          prediction model exceeds the VRAM budget; layers
                          spill to system RAM via HuggingFace accelerate.

The four windows are sequenced by the bench orchestrator (LATTICE-class
component); nvinx provides placement decisions for each window independently.
"""

from nvinx.catalog import HardwareSpec, ModelSpec, Residency
from nvinx.patterns import fractional_coresidency, ram_overflow, serial_handoff

# Bench hardware envelope. Representative of an entry-class GPU workstation
# (e.g., 4 GB-class consumer or workstation GPU, 32 GB system RAM, 8-core CPU).
BENCH = HardwareSpec(vram_gb=4.0, ram_gb=32.0, cpu_cores=8)


# ──────────────────────────────────────────────────────────────────────────────
# Window 1 — overnight basecalling (Pattern A: serial_handoff)
# ──────────────────────────────────────────────────────────────────────────────


def window_overnight_basecalling():
    """The basecaller saturates the GPU for 4-8 hours. CPU work runs in parallel."""
    print("\n--- Window 1: overnight basecalling (Pattern A) ---")

    basecaller = ModelSpec(
        name="long_read_basecaller",
        vram_gb=3.0,
        residency=Residency.GPU_EXCLUSIVE,
    )

    cpu_during_window = [
        # Post-basecall variant calling (CPU-only by design)
        ModelSpec(name="variant_caller", vram_gb=0.0, residency=Residency.CPU_ONLY),
        # QC report generation
        ModelSpec(name="qc_reporter", vram_gb=0.0, residency=Residency.CPU_ONLY),
        # Case-record scaffolding (document I/O)
        ModelSpec(name="case_record_scaffold", vram_gb=0.0, residency=Residency.CPU_ONLY),
        # Pre-fetch literature for the typed HLA set
        ModelSpec(name="literature_prefetch", vram_gb=0.0, residency=Residency.CPU_ONLY),
    ]

    plan = serial_handoff(basecaller, cpu_during_window, BENCH)
    print(f"  GPU foreground : {plan.gpu_foreground.name}")
    print(f"  CPU parallel   : {[m.name for m in plan.cpu_parallel]}")
    if plan.unscheduled:
        print(f"  unscheduled    : {[m.name for m in plan.unscheduled]}")


# ──────────────────────────────────────────────────────────────────────────────
# Window 2 — neoantigen ranking ensemble (Pattern B: fractional_coresidency)
# ──────────────────────────────────────────────────────────────────────────────


def window_neoantigen_ensemble():
    """Two small inference models co-reside in VRAM; a CPU lookup runs alongside."""
    print("\n--- Window 2: neoantigen ranking (Pattern B) ---")

    candidates = [
        # Mutation-class classifier (small protein language model + LoRA head)
        ModelSpec(
            name="mutation_classifier",
            vram_gb=0.5,
            residency=Residency.GPU_SHARED,
        ),
        # MHC-I binding affinity predictor
        ModelSpec(
            name="mhc1_binding_predictor",
            vram_gb=0.2,
            residency=Residency.GPU_SHARED,
            cpu_fallback_supported=True,  # can fall back to CPU for novel alleles
        ),
        # Shared-antigen library lookup (CPU only)
        ModelSpec(
            name="shared_antigen_lookup",
            vram_gb=0.0,
            residency=Residency.CPU_ONLY,
        ),
    ]

    plan = fractional_coresidency(candidates, BENCH)
    print(f"  GPU co-resident: {[m.name for m in plan.gpu_coresident]}")
    print(f"  CPU parallel   : {[m.name for m in plan.cpu_parallel]}")
    print(f"  notes          : {plan.notes}")


# ──────────────────────────────────────────────────────────────────────────────
# Window 3 — structure prediction, fits in VRAM (sequential, no nvinx pattern)
# ──────────────────────────────────────────────────────────────────────────────


def window_structure_prediction_in_vram():
    """The standard case: structure prediction for ≤216-residue candidates fits."""
    print("\n--- Window 3: structure prediction, fits in VRAM ---")

    folder = ModelSpec(
        name="structure_predictor",
        vram_gb=2.2,
        residency=Residency.GPU_EXCLUSIVE,
    )
    # No co-residents during this exclusive window. nvinx is not invoked
    # here because there is no scheduling decision — the orchestrator simply
    # runs the model exclusively. Pattern A could be invoked with no CPU
    # candidates if the orchestrator wants a uniform interface.
    print(f"  GPU foreground : {folder.name} ({folder.vram_gb} GB, exclusive)")


# ──────────────────────────────────────────────────────────────────────────────
# Window 4 — oversized candidate (>216 residues): RAM overflow (Pattern C)
# ──────────────────────────────────────────────────────────────────────────────


def window_structure_prediction_overflow():
    """When a candidate exceeds the residue threshold, the model exceeds VRAM.

    Layers spill to system RAM via HuggingFace accelerate's device_map='auto'.
    The 3-5x slowdown is preferable to OOMing the bench.
    """
    print("\n--- Window 4: structure prediction with RAM overflow (Pattern C) ---")

    big_folder = ModelSpec(
        name="structure_predictor_oversized",
        vram_gb=6.0,
        residency=Residency.GPU_RAM_OVERFLOW,
        ram_overflow_supported=True,
        ram_gb_needed=10.0,
    )
    hint = ram_overflow(big_folder, BENCH)
    print(f"  device_map     : {hint['device_map']}")
    print(f"  max_memory     : {hint['max_memory']}")
    print(f"  est. slowdown  : {hint['estimated_slowdown']}")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    print(f"Bench: VRAM={BENCH.vram_gb} GB, RAM={BENCH.ram_gb} GB, CPU={BENCH.cpu_cores}c")
    print("Workload: bioinformatics + neoantigen ranking + structure prediction")
    print("All four windows are scheduled with framework-agnostic primitives;")
    print("the runtime is the user's choice (PyTorch, HuggingFace, etc.).")

    window_overnight_basecalling()
    window_neoantigen_ensemble()
    window_structure_prediction_in_vram()
    window_structure_prediction_overflow()


if __name__ == "__main__":
    main()
