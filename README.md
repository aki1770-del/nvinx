# nvinx

[![CI](https://github.com/aki1770-del/nvinx/actions/workflows/ci.yml/badge.svg)](https://github.com/aki1770-del/nvinx/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nvinx.svg)](https://pypi.org/project/nvinx/)
[![Python](https://img.shields.io/pypi/pyversions/nvinx.svg)](https://pypi.org/project/nvinx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Your 4 GB GPU is not idle during a long inference — it's a scheduling constraint.**

When a single model saturates a small GPU for hours, the CPU, system RAM, and the wall clock are still available. Most research tooling doesn't schedule against those idle resources — it just waits. `nvinx` is three small, composable patterns that turn "wait" into "work" on limited-VRAM benches.

*Extracted from a real single-GPU research workload (4 GB VRAM + 32 GB RAM + 8 cores). The patterns are generic; the workload is not published.*

**Status:** v0.1 alpha. API may still change between minor versions; pin your dependency.

---

## The reframe

A research bench with one small GPU has more compute than it looks like. During a 6-hour basecaller or folding run, the GPU is saturated — but the CPU, system RAM, and I/O channels are not. The usual response is to treat that window as a blocker and run the remaining steps serially afterward, doubling total wall-clock.

The reframe: the GPU-exclusive window is a **scheduling opportunity**, not a dead time. CPU-only work (variant calling, report generation, data ingest, classical-ML inference) can run in parallel for free. Small-VRAM models can co-reside with each other. Models that exceed VRAM can spill to RAM via HuggingFace `accelerate` with a known 3–5× slowdown — which is often strictly better than not running at all.

`nvinx` packages three named patterns for these cases. It is pure scheduling logic: no model invocation, no memory reservation, no runtime. You call a pattern with `ModelSpec` inputs and a `HardwareSpec` envelope, you get a `SchedulingPlan` out, and your runtime executes the plan. This keeps `nvinx` testable, framework-agnostic, and composable with whatever runtime you already have.

---

## Install

```bash
pip install nvinx
```

From source:

```bash
git clone https://github.com/aki1770-del/nvinx
cd nvinx
pip install -e ".[dev]"
pytest
```

Python ≥ 3.10. Dependencies: `pyyaml` only.

---

## The three patterns

### Pattern A — `serial_handoff`: CPU work during a GPU-exclusive window

**When:** one long GPU job saturates the card, and you have other work that doesn't need the GPU.

**Why:** the GPU-exclusive window is the CPU's foreground production window. Running CPU-only work sequentially *after* the GPU job wastes wall-clock.

```python
from nvinx.catalog import HardwareSpec, ModelSpec, Residency
from nvinx.patterns import serial_handoff

hw = HardwareSpec(vram_gb=4.0, ram_gb=32.0, cpu_cores=8)

gpu_job = ModelSpec(
    name="long_inference",
    vram_gb=3.5,
    residency=Residency.GPU_EXCLUSIVE,
)

cpu_candidates = [
    ModelSpec(name="feature_extractor",  vram_gb=0.0, residency=Residency.CPU_ONLY),
    ModelSpec(name="report_writer",      vram_gb=0.0, residency=Residency.CPU_ONLY),
    ModelSpec(name="vector_index_build", vram_gb=0.0, residency=Residency.CPU_ONLY),
]

plan = serial_handoff(gpu_job, cpu_candidates, hw)

print(plan.gpu_foreground.name)         # "long_inference"
print([m.name for m in plan.cpu_parallel])
# ['feature_extractor', 'report_writer', 'vector_index_build']
```

The pattern refuses to run if the GPU job isn't marked `GPU_EXCLUSIVE` or exceeds the hardware's VRAM. CPU candidates without `cpu_fallback_supported=True` (and not `CPU_ONLY`) land in `plan.unscheduled`.

### Pattern B — `fractional_coresidency`: pack small-VRAM models into a shared budget

**When:** multiple models each small enough to fit alongside others (e.g. a classifier + an embedder + a re-ranker).

**Why:** modern research stacks often chain several small models. Running them sequentially on a shared GPU means repeated load/unload cycles. Packing them into one co-resident window avoids the churn and cuts pipeline latency.

```python
from nvinx.patterns import fractional_coresidency

candidates = [
    ModelSpec(name="classifier_a", vram_gb=0.7, residency=Residency.GPU_SHARED),
    ModelSpec(name="embedder",     vram_gb=0.3, residency=Residency.GPU_SHARED),
    ModelSpec(name="reranker",     vram_gb=1.2, residency=Residency.GPU_SHARED),
    ModelSpec(
        name="oversized_model",
        vram_gb=4.0,
        residency=Residency.GPU_SHARED,
        cpu_fallback_supported=True,
    ),
]

plan = fractional_coresidency(candidates, hw)

print([m.name for m in plan.gpu_coresident])
# ['reranker', 'classifier_a', 'embedder']  # packed 2.2 / 3.5 GB after headroom

print([m.name for m in plan.cpu_parallel])
# ['oversized_model']  # falls back to CPU because it doesn't fit
```

Greedy bin-packing by descending `vram_gb`. The budget is `hw.vram_gb − headroom_gb` (default 0.5 GB for activations and kernel launches). `GPU_EXCLUSIVE` models are never co-resident; `CPU_ONLY` models always land in `cpu_parallel`.

### Pattern C — `ram_overflow`: spill layers from VRAM to system RAM

**When:** one model exceeds the VRAM budget but system RAM is plentiful, and the model supports layer offloading.

**Why:** a 6 GB model on a 4 GB card normally OOMs on load. With HuggingFace `accelerate`'s `device_map="auto"`, the layers that don't fit in VRAM run from system RAM. The tradeoff is a 3–5× slowdown — which is often strictly better than not running the model at all.

```python
from nvinx.patterns import ram_overflow

big_model = ModelSpec(
    name="big_folder",
    vram_gb=6.0,
    residency=Residency.GPU_RAM_OVERFLOW,
    ram_overflow_supported=True,
    ram_gb_needed=10.0,
)

hint = ram_overflow(big_model, hw)

print(hint["device_map"])   # "auto"
print(hint["max_memory"])   # {0: '3.5GiB', 'cpu': '6.5GiB'}
print(hint["estimated_slowdown"])
# "3-5x vs. pure-GPU inference"

# Pass through to your runtime:
# from transformers import AutoModelForXxx
# AutoModelForXxx.from_pretrained(
#     "model-name",
#     device_map=hint["device_map"],
#     max_memory=hint["max_memory"],
# )
```

`ram_overflow` returns a hint dict (not a `SchedulingPlan`) because the runtime contract is different — you're passing keyword arguments to `from_pretrained`, not placing multiple models. This is a deliberate asymmetry.

---

## Which pattern do I need?

| Situation | Pattern |
|---|---|
| One big GPU job running for hours + other work that doesn't need the GPU | **A** `serial_handoff` |
| Several models that each fit in VRAM alongside others | **B** `fractional_coresidency` |
| One model that exceeds VRAM on load | **C** `ram_overflow` |
| Mix: long GPU job + multiple small follow-on models on the same GPU | **A** for the long job, then **B** on the follow-on window |
| Model that doesn't fit *and* no RAM headroom | None — buy more RAM, quantize, or switch models |

The three patterns compose: a realistic day on a 4 GB bench runs Pattern A during a nightly basecaller, Pattern B across a morning of small-model inference, and Pattern C whenever a large folding model is needed. `nvinx` doesn't orchestrate the transitions — it gives you the placement decisions one at a time.

---

## Data model

Four types live in `src/nvinx/catalog.py`:

- `HardwareSpec(vram_gb, ram_gb, cpu_cores)` — the physical envelope.
- `ModelSpec(name, vram_gb, residency, cpu_fallback_supported, ram_overflow_supported, ram_gb_needed)` — a workload to schedule.
- `Residency` — enum: `GPU_EXCLUSIVE`, `GPU_SHARED`, `CPU_ONLY`, `GPU_RAM_OVERFLOW`.
- `SchedulingPlan` — the output: `gpu_foreground`, `gpu_coresident`, `cpu_parallel`, `overflow`, `unscheduled`, `notes`.

Every pattern takes `ModelSpec`s and a `HardwareSpec`, returns a `SchedulingPlan` (or a hint dict, for Pattern C). The pattern functions are pure; they never touch the GPU.

---

## What `nvinx` is not

- **Not a runtime.** `nvinx` returns placement decisions. You execute them with whatever runtime you already use (PyTorch, HuggingFace `transformers`, vLLM, custom).
- **Not a job queue.** Pattern A assumes you already know which job is GPU-exclusive *right now*. Higher-level orchestration (what runs first? what triggers the next window?) is out of scope.
- **Not a profiler.** `ModelSpec.vram_gb` is *your* declaration of the model's VRAM footprint. `nvinx` trusts it. If you don't know the footprint, measure it first; then call `nvinx`.
- **Not NVIDIA- or NGINX-specific.** The name is a portmanteau. See the disclaimer below.

---

## Status and stability

v0.1 is **alpha**. Expect:

- Additions: configs/YAML-driven spec loading (v0.2), a `scheduler` layer that composes patterns across a day's workload (v0.2+), more patterns as edge cases surface.
- Breaking changes: `SchedulingPlan` shape may add fields. Pin your version.
- No breaking changes to the three core pattern signatures — those are stable for v0.1.

---

## Contributing — case-study YAMLs wanted

The patterns were derived from one real research workload. They'll become more robust if other people throw their workloads at them and tell us what breaks.

The highest-value contribution right now is **a case-study YAML describing your bench**:

- What hardware? (VRAM, RAM, cores)
- Which models? (name, VRAM footprint, whether they support CPU fallback / RAM overflow)
- Which pattern did you try, and what happened?
- What was the old wall-clock, and what did `nvinx` change?

Submit as a pull request to `examples/case_studies/` (or open an issue with the YAML inline). No anonymization requirement — feel free to keep your workload names vague if they're sensitive.

Also welcome:

- Bug reports with a minimal reproduction
- New pattern proposals (open an issue first so we can discuss scope before you implement)

Star the repo if you find the reframe useful — it's the signal that tells us to keep extracting patterns from the parent workload.

---

## Running the tests and linters

```bash
pip install -e ".[dev]"

pytest              # 8 smoke tests
ruff format --check .
ruff check .
```

Eight smoke tests covering the three patterns' happy paths, error paths, and boundary cases. Tests live in `tests/test_patterns.py`. CI runs on every push and PR across Python 3.10–3.12.

---

## License

MIT. See [LICENSE](LICENSE).

---

## Trademark disclaimer

*`nvinx` is not affiliated with NVIDIA Corporation or NGINX Inc. / F5, Inc. The name is a portmanteau reflecting the nginx-inspired upstream-routing metaphor applied to heterogeneous compute. If trademark concerns surface, the fallback package name is `nvginx`.*
