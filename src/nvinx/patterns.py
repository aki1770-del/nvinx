"""
Three heterogeneous-compute scheduling patterns for limited-VRAM GPU benches.

All patterns are pure: they do not invoke models, reserve memory, or touch
the GPU. They accept ModelSpec / HardwareSpec inputs and return a
SchedulingPlan. A runtime is expected to consume the plan and execute it.

Pattern  Trigger                                    Output behavior
-------  -----------------------------------------  -----------------------------------
A        One GPU-exclusive job saturates the card   Run CPU-only work in parallel
B        Multiple small-VRAM models fit together    Pack into the VRAM budget
C        A single model exceeds the VRAM budget     Offload layers to system RAM
"""

from __future__ import annotations

from collections.abc import Mapping

from nvinx.catalog import HardwareSpec, ModelSpec, Residency, SchedulingPlan
from nvinx.interference import (
    HardwareCoefficients,
    InterferenceProfile,
    PairLookupEntry,
    asymmetry_predictor,
    max_kernel_rate_score,
    predict_pair_latency,
)

_DEFAULT_HEADROOM_GB = 0.5


def serial_handoff(
    gpu_exclusive: ModelSpec,
    cpu_candidates: list[ModelSpec],
    hw: HardwareSpec,
) -> SchedulingPlan:
    """Pattern A — run CPU work during a GPU-exclusive window.

    When one workload saturates the GPU (e.g. a long basecalling or folding
    run), CPU-only workloads can execute in parallel without contention. The
    GPU-exclusive window is not idle wall-clock — it is the CPU's foreground
    production window.

    Parameters
    ----------
    gpu_exclusive
        The GPU-saturating workload. Must have residency=GPU_EXCLUSIVE.
    cpu_candidates
        Workloads eligible to run on CPU during the GPU window.
    hw
        Physical envelope (used for sanity checks).

    Returns
    -------
    SchedulingPlan
        ``gpu_foreground`` holds the exclusive job; ``cpu_parallel`` holds the
        CPU-eligible candidates; ``unscheduled`` holds candidates that cannot
        run in this window (e.g. they need the GPU and have no CPU fallback).

    Raises
    ------
    ValueError
        If ``gpu_exclusive`` is not marked GPU_EXCLUSIVE or exceeds hw.vram_gb.
    """
    if gpu_exclusive.residency is not Residency.GPU_EXCLUSIVE:
        raise ValueError(
            f"serial_handoff requires residency=GPU_EXCLUSIVE; "
            f"got {gpu_exclusive.residency} for {gpu_exclusive.name}"
        )
    if gpu_exclusive.vram_gb > hw.vram_gb:
        raise ValueError(
            f"{gpu_exclusive.name} requires {gpu_exclusive.vram_gb} GB VRAM "
            f"but hardware has only {hw.vram_gb} GB"
        )

    cpu_parallel: list[ModelSpec] = []
    unscheduled: list[ModelSpec] = []
    for candidate in cpu_candidates:
        if candidate.residency is Residency.CPU_ONLY or candidate.cpu_fallback_supported:
            cpu_parallel.append(candidate)
        else:
            unscheduled.append(candidate)

    return SchedulingPlan(
        gpu_foreground=gpu_exclusive,
        cpu_parallel=cpu_parallel,
        unscheduled=unscheduled,
        notes=[
            f"Pattern A (serial_handoff): GPU held by {gpu_exclusive.name}; "
            f"{len(cpu_parallel)} CPU workload(s) run in parallel."
        ],
    )


def fractional_coresidency(
    candidates: list[ModelSpec],
    hw: HardwareSpec,
    *,
    headroom_gb: float = _DEFAULT_HEADROOM_GB,
) -> SchedulingPlan:
    """Pattern B — pack multiple small-VRAM models into a shared budget.

    Greedy bin-packing: sort candidates by ``vram_gb`` descending and pack
    until the budget (``hw.vram_gb`` minus ``headroom_gb``) is reached.
    Candidates that don't fit fall back to CPU if supported; otherwise they
    are returned as unscheduled. GPU_EXCLUSIVE candidates are never
    co-resident.

    Parameters
    ----------
    candidates
        Workloads to consider.
    hw
        Physical envelope.
    headroom_gb
        VRAM reserved for activations and kernel launches; default 0.5 GB.

    Returns
    -------
    SchedulingPlan
        ``gpu_coresident`` holds packed models; ``cpu_parallel`` holds CPU
        fallbacks; ``unscheduled`` holds candidates that cannot run.

    Raises
    ------
    ValueError
        If the budget after headroom is non-positive.
    """
    budget = hw.vram_gb - headroom_gb
    if budget <= 0:
        raise ValueError(
            f"VRAM budget after {headroom_gb} GB headroom is non-positive (vram_gb={hw.vram_gb})"
        )

    sorted_candidates = sorted(candidates, key=lambda m: m.vram_gb, reverse=True)

    gpu_coresident: list[ModelSpec] = []
    cpu_parallel: list[ModelSpec] = []
    unscheduled: list[ModelSpec] = []
    used_vram = 0.0

    for candidate in sorted_candidates:
        if candidate.residency is Residency.GPU_EXCLUSIVE:
            unscheduled.append(candidate)
            continue
        if candidate.residency is Residency.CPU_ONLY:
            cpu_parallel.append(candidate)
            continue
        if used_vram + candidate.vram_gb <= budget:
            gpu_coresident.append(candidate)
            used_vram += candidate.vram_gb
        elif candidate.cpu_fallback_supported:
            cpu_parallel.append(candidate)
        else:
            unscheduled.append(candidate)

    return SchedulingPlan(
        gpu_coresident=gpu_coresident,
        cpu_parallel=cpu_parallel,
        unscheduled=unscheduled,
        notes=[
            f"Pattern B (fractional_coresidency): {len(gpu_coresident)} model(s) "
            f"co-resident; {used_vram:.2f}/{budget:.2f} GB VRAM used "
            f"(+ {headroom_gb} GB headroom)."
        ],
    )


def fractional_coresidency_v2(
    candidates: list[ModelSpec],
    hw: HardwareSpec,
    *,
    headroom_gb: float = _DEFAULT_HEADROOM_GB,
    interference_profiles: Mapping[str, InterferenceProfile] | None = None,
    hw_coefs: HardwareCoefficients | None = None,
    pair_lookup: Mapping[tuple[str, str], PairLookupEntry] | None = None,
    max_kernel_rate_threshold: float | None = None,
    gamma_kernel_size: float | None = None,
) -> SchedulingPlan:
    """Pattern B v0.2/v0.3 — queue-aware interference-modeled co-residency.

    Backward-compatible enhancement of v0.1 ``fractional_coresidency``. Calls
    the v0.1 greedy bin-packer for the placement decision, then augments the
    plan's ``notes`` with interference predictions if interference_profiles
    are provided.

    The placement DECISION itself is unchanged from v0.1 (greedy bin-pack by
    descending VRAM). The v0.2/v0.3 additions are diagnostic + advisory:

      - max_kernel_rate pre-filter (heuristic; ρ=0.50 with slowdown)
      - per-pair latency prediction (queue-aware formula or lookup)
      - asymmetry predictor (act_solo_ratio; ρ=0.72)

    Interference profiles are operator-generated; the package ships no
    published reference profile data. See ``docs/calibrating-your-substrate.md``
    for the operator workflow. Calibration tooling itself is research-shape
    and lives in a separate workspace (private); a turnkey
    ``nvinx.calibration`` module is a v0.3+ goal.

    Parameters
    ----------
    candidates, hw, headroom_gb
        Same as v0.1 ``fractional_coresidency``.
    interference_profiles
        Mapping from model name to InterferenceProfile. If None, this function
        is equivalent to v0.1 ``fractional_coresidency`` (no augmentation).
    hw_coefs
        Substrate-level coefficients (idlef + powerp). Required for
        queue-aware prediction; if None, profiles must rely on lookup-only.
    pair_lookup
        Per-pair measured ground truth (safety net for known high-error
        pairs like 2-small-kernel cases per v0.3 D2 outlier finding).
    max_kernel_rate_threshold
        If set, emit a warning note when max kernel rate of placed models
        exceeds this value (high queue-contention risk).
    gamma_kernel_size
        Optional V5 kernel-size-ratio coefficient (v0.6 alpha; substrate-
        bound). When supplied, per-pair predictions route through the V5
        formula via :func:`~nvinx.interference.predict_pair_latency_queue_aware_v5`;
        when ``None`` (default), v0.2 queue-aware behaviour is preserved.
        Operator-fit via :func:`~nvinx.interference.fit_gamma_kernel_size`.

    Returns
    -------
    SchedulingPlan
        Same placement as v0.1; ``notes`` augmented with interference info.
    """
    base_plan = fractional_coresidency(candidates, hw, headroom_gb=headroom_gb)

    if interference_profiles is None:
        return base_plan

    placed = base_plan.gpu_coresident
    placed_with_profiles = [m for m in placed if m.name in interference_profiles]
    notes = list(base_plan.notes)

    # Heuristic pre-filter: max kernel rate
    if placed_with_profiles:
        profiles = [interference_profiles[m.name] for m in placed_with_profiles]
        mkr = max_kernel_rate_score(profiles)
        notes.append(
            f"interference: max_kernel_rate={mkr:.1f} k/ms "
            f"({len(profiles)}/{len(placed)} placed models have profiles)"
        )
        if max_kernel_rate_threshold is not None and mkr > max_kernel_rate_threshold:
            notes.append(
                f"interference: WARNING max_kernel_rate {mkr:.1f} > "
                f"threshold {max_kernel_rate_threshold:.1f}; expect queue contention"
            )

    # Per-pair prediction for diagnostics (only if hw_coefs OR pair_lookup available)
    if (hw_coefs is not None or pair_lookup is not None) and len(placed_with_profiles) >= 2:
        for i in range(len(placed_with_profiles)):
            for j in range(i + 1, len(placed_with_profiles)):
                a = placed_with_profiles[i]
                b = placed_with_profiles[j]
                pa = interference_profiles[a.name]
                pb = interference_profiles[b.name]
                # Use a sentinel HardwareCoefficients if hw_coefs is None
                # (queue-aware path will fail gracefully via fallback_unknown)
                if hw_coefs is not None:
                    lat_a, lat_b, source = predict_pair_latency(
                        pa,
                        pb,
                        hw_coefs,
                        pair_lookup=pair_lookup,
                        gamma_kernel_size=gamma_kernel_size,
                    )
                    asym = asymmetry_predictor(pa, pb)
                    notes.append(
                        f"interference: pair({a.name}+{b.name}) "
                        f"pred_lat=({lat_a:.1f}, {lat_b:.1f})ms via {source}; "
                        f"asymmetry={asym:.2f}"
                    )
                elif pair_lookup is not None:
                    # Lookup-only path
                    canonical = tuple(sorted([a.name, b.name]))
                    entry = pair_lookup.get(canonical)
                    if entry is not None:
                        notes.append(
                            f"interference: pair({a.name}+{b.name}) "
                            f"meas_lat=({entry.measured_latency_a_ms:.1f}, "
                            f"{entry.measured_latency_b_ms:.1f})ms via lookup"
                        )

    return SchedulingPlan(
        gpu_coresident=base_plan.gpu_coresident,
        cpu_parallel=base_plan.cpu_parallel,
        unscheduled=base_plan.unscheduled,
        notes=notes,
    )


def ram_overflow(
    model: ModelSpec,
    hw: HardwareSpec,
    *,
    headroom_gb: float = _DEFAULT_HEADROOM_GB,
) -> dict[str, object]:
    """Pattern C — spill layers from VRAM to system RAM.

    When a model's footprint exceeds the VRAM budget but system RAM is
    available, layers can be offloaded via HuggingFace ``accelerate``'s
    ``device_map="auto"``. The tradeoff is a 3–5× slowdown versus pure-GPU
    inference, but the job runs at all instead of OOMing on load.

    This function returns a hint dict; the runtime is expected to pass the
    relevant keys to ``from_pretrained(..., device_map=..., max_memory=...)``.

    Parameters
    ----------
    model
        A workload that exceeds ``hw.vram_gb``. Must have
        ``ram_overflow_supported=True`` and a declared ``ram_gb_needed``.
    hw
        Physical envelope.
    headroom_gb
        VRAM reserved for activations; default 0.5 GB.

    Returns
    -------
    dict
        Keys: ``device_map``, ``max_memory``, ``estimated_slowdown``, ``notes``.

    Raises
    ------
    ValueError
        If the model does not declare ram_overflow support or the RAM
        requirement exceeds hardware.
    """
    if not model.ram_overflow_supported:
        raise ValueError(f"{model.name} does not declare ram_overflow_supported=True")
    if model.ram_gb_needed is None:
        raise ValueError(
            f"{model.name} has ram_overflow_supported=True but no ram_gb_needed declared"
        )
    if model.ram_gb_needed > hw.ram_gb:
        raise ValueError(
            f"{model.name} needs {model.ram_gb_needed} GB RAM but hardware has only {hw.ram_gb} GB"
        )

    gpu_budget_gb = max(0.0, hw.vram_gb - headroom_gb)
    cpu_budget_gb = max(0.0, model.ram_gb_needed - gpu_budget_gb)

    return {
        "device_map": "auto",
        "max_memory": {
            0: f"{gpu_budget_gb:.1f}GiB",
            "cpu": f"{cpu_budget_gb:.1f}GiB",
        },
        "estimated_slowdown": "3-5x vs. pure-GPU inference",
        "notes": (
            f"Pattern C (ram_overflow): {model.name} split across GPU "
            f"({gpu_budget_gb:.1f} GB) and system RAM ({cpu_budget_gb:.1f} GB)."
        ),
    }
