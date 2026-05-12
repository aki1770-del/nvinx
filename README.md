# nvinx

[![CI](https://github.com/aki1770-del/nvinx/actions/workflows/ci.yml/badge.svg)](https://github.com/aki1770-del/nvinx/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nvinx.svg)](https://pypi.org/project/nvinx/)
[![Python](https://img.shields.io/pypi/pyversions/nvinx.svg)](https://pypi.org/project/nvinx/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Your 4 GB GPU is not idle during a long inference — it's a scheduling constraint.**

When a single model saturates a small GPU for hours, the CPU, system RAM, and the wall clock are still available. Most research tooling doesn't schedule against those idle resources — it just waits. `nvinx` is three small, composable patterns that turn "wait" into "work" on limited-VRAM benches.

*Extracted from a real single-GPU research workload (4 GB VRAM + 32 GB RAM + 8 cores). The patterns are generic; the workload is not published.*

**Status:** v0.2.0a1 in-tree (v0.1.0 on PyPI). API may still change between minor versions; pin your dependency.

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

## v0.2 (in-tree): interference prediction for Pattern B

Pattern B v0.1 packs models that fit. v0.2 adds **interference prediction primitives** for operators who want to know whether a packed placement will hit SLO *before* they run it.

**The pain v0.2 addresses.** Pattern B v0.1 says yes/no on packing — but doesn't tell you whether the packed models will interfere. Two small models that fit in VRAM may still slow each other down 3-5× under co-residency due to GPU kernel-queue contention. The operator wants:

- *"If I pack model A and model B, what latency should I expect?"*
- *"Is this placement near the SLO bound?"*
- *"Which model will suffer more if they co-reside?"*

**v0.2 solution (substrate-native).** A new module `nvinx.interference` provides:

- `InterferenceProfile` — per-model coefficients (operator-profiled on *your* substrate)
- `HardwareCoefficients` — substrate-level coefficients (one-time per bench)
- `predict_pair_latency_queue_aware()` — queue-aware formula: `latency_i = act_solo_i × (1 + θ_i × partner_act / (act_solo_i + partner_act)) + scheduling_delay`
- `max_kernel_rate_score()` — pre-filter heuristic (Spearman ρ ≈ 0.50 with measured slowdown on the validation bench)
- `asymmetry_predictor()` — `act_solo_ratio` for which-suffers-more (ρ ≈ 0.72)
- `predict_pair_latency()` — tiered: lookup → queue-aware → fallback
- `PairLookupEntry` — per-pair measured ground truth (safety net for known high-error pairs)

`fractional_coresidency_v2()` accepts these as optional inputs and augments the plan's `notes` with predictions. If you don't supply profiles, it's equivalent to v0.1 `fractional_coresidency` (no behaviour change).

**Honest scope.** The queue-aware formula was validated on **one substrate**: a 4 GB RTX A1000 mobile bench (Ampere sm_86) with a heterogeneous transformer/LLM workload mix (ESM-2-150M at 3 sequence lengths + Qwen-0.5B + Whisper-base). On that substrate's 4-model 6-pair corpus, the formula achieves ~16% LOPO mean error; on the 5-model 10-pair extended corpus, ~25% LOPO mean. Persistent ~30% LOPO outlier on 2-small-kernel pairs (formula limit; lookup safety net handles those).

**Substrate-bound.** If your bench differs (different GPU, different model class, different driver), the published coefficients are not portable — you must recalibrate. See [`docs/calibrating-your-substrate.md`](docs/calibrating-your-substrate.md) for the operator workflow.

**Code example.**

```python
from nvinx.catalog import HardwareSpec, ModelSpec, Residency
from nvinx.interference import HardwareCoefficients, InterferenceProfile
from nvinx.patterns import fractional_coresidency_v2

hw = HardwareSpec(vram_gb=4.0, ram_gb=32.0, cpu_cores=20)

# Substrate-level coefficients (you fit these once via the calibration workflow)
hw_coefs = HardwareCoefficients(
    idlef_polynomial=(6.42, -7.0),
    powerp_linear=(0.0,),
    nominal_freq_mhz=1530.0,
    tdp_watts=40.0,
    substrate_name="rtx_a1000_4gb",
)

# Per-model coefficients (you profile these for each model on your substrate)
profile_a = InterferenceProfile(
    name="model_a",
    kernels=1027, baseidle_ms=0.077, act_solo_ms=21.7,
    l2_saturation_pct=16.8, theta=3.91,
    architecture_class="encoder_transformer",
)
profile_b = InterferenceProfile(
    name="model_b",
    kernels=1026, baseidle_ms=0.149, act_solo_ms=114.9,
    l2_saturation_pct=37.1, theta=1.20,
    architecture_class="encoder_transformer",
)

candidates = [
    ModelSpec(name="model_a", vram_gb=0.6, residency=Residency.GPU_SHARED),
    ModelSpec(name="model_b", vram_gb=0.6, residency=Residency.GPU_SHARED),
]

plan = fractional_coresidency_v2(
    candidates, hw,
    interference_profiles={"model_a": profile_a, "model_b": profile_b},
    hw_coefs=hw_coefs,
    max_kernel_rate_threshold=50.0,
)

for note in plan.notes:
    print(note)
# Pattern B (fractional_coresidency): 2 model(s) co-resident; 1.20/3.50 GB VRAM used (+ 0.5 GB headroom).
# interference: max_kernel_rate=47.3 k/ms (2/2 placed models have profiles)
# interference: pair(model_a+model_b) pred_lat=(86.5, 141.7)ms via queue_aware; asymmetry=5.30
```

The placement *decision* is unchanged from v0.1 (greedy bin-pack). The v0.2 additions are diagnostic + advisory — they help you decide whether to accept the placement.

---

## v0.3 (in-tree, alpha): substrate-bound V5 kernel-size-ratio correction

v0.3 adds **one optional empirical term** to the v0.2 queue-aware formula. On substrates where the dominant interference mechanism is the GPU scheduler queue rather than L2 cache pressure, the term captures the asymmetric blocking of small-kernel models by large-kernel partners — and reduces LOPO mean error by ~5 pp on transformer/LLM workload mixes on a 4 GB mobile substrate.

**The pain v0.3 addresses.** v0.2's queue-aware formula treats both sides of a co-resident pair symmetrically in the partner-time-fraction term. On heterogeneous-kernel-size corpora (e.g. a decoder LLM with many small kernels co-located with an encoder model with fewer larger kernels), the symmetric form under-predicts the slowdown on the small-kernel side: every small-self kernel may queue behind a long-running large-partner kernel. The asymmetric blocking is not captured by `partner_frac` alone — it depends on the *kernel-duration ratio* of each side.

**v0.3 solution (V5; substrate-bound).** A new function `nvinx.interference.predict_pair_latency_queue_aware_v5` extends v0.2 with one additional scalar `gamma_kernel_size`:

```
partner_frac      = act_partner / (act_self + act_partner)
kernel_size_ratio = (act_partner / kernels_partner) / (act_self / kernels_self)
latency_self      = act_self × (1 + theta_self × partner_frac
                                  × (1 + gamma × kernel_size_ratio))
                  + scheduling_delay + baseidle_self
```

A companion function `nvinx.interference.fit_gamma_kernel_size(profiles, pair_measurements, hw)` solves the closed-form relative-residual weighted least-squares fit (no `numpy` / `scipy` dependency added; pure stdlib). The dispatcher `predict_pair_latency` and `fractional_coresidency_v2` both accept an optional `gamma_kernel_size` kwarg and route to V5 when supplied.

**Backward compatibility (the load-bearing discipline).** When `gamma_kernel_size` is `None` (the default) or `0.0`, the V5 function is **bit-identical** to v0.2's `predict_pair_latency_queue_aware`. Existing v0.2 callers see no behaviour change. The `predict_pair_latency` dispatcher's source label remains `"queue_aware"` when no γ is supplied; it returns `"queue_aware_v5"` only when γ is opted in.

**Honest scope (when V5 helps vs hurts).**

- *V5 helps* on heterogeneous-kernel-size corpora on small-SM substrates where queue contention dominates (the reference bench has 16 SMs).
- *V5 may not help — or may hurt* on datacenter-class GPUs with many SMs (Volta / Ampere / Hopper datacenter; 80+ SMs) where the SM pool absorbs multiple co-located workloads and L2 cache contention is the dominant interference physics. The published `iGniter` evaluation on a Volta-class datacenter GPU + CNN saw max ~1.35× slowdown at 5 co-located workloads; the reference bench saw max ~12.11× slowdown at 2 workloads. The ~30× magnitude gap is the empirical signature of fundamentally different interference physics — V5's reference γ ≈ 0.75 will be wrong by an unknown sign on that substrate class.
- *V5 collapses to v0.2* on same-architecture corpora where `kernel_size_ratio ≈ 1`.

**Substrate-bound — the reference γ does NOT transfer.** The reference value `γ ≈ 0.75` cited in the docstrings was fitted on an extended 7-model 19-pair corpus on a 4 GB RTX A1000 mobile substrate with a transformer/LLM workload mix. Earlier iterations of the same calibration journey: a 6-model 15-pair fit yielded γ ≈ 0.44, and a 4-model 6-pair fit yielded γ → 0 (the asymmetry signal does not appear with too few or too uniform models in the corpus). **Operators on other substrates MUST refit** via `fit_gamma_kernel_size` on their own calibration data; see [`docs/calibrating-your-substrate.md`](docs/calibrating-your-substrate.md) Step 6 for the workflow.

**Code example.**

```python
from nvinx import (
    HardwareCoefficients,
    InterferenceProfile,
    fit_gamma_kernel_size,
    fractional_coresidency_v2,
)

# Steps 1-5 of docs/calibrating-your-substrate.md produced these:
hw_coefs: HardwareCoefficients = ...
profiles: dict[str, InterferenceProfile] = ...  # theta fitted per model

# Cross-pair co-located measurements (≥ 6 pairs spanning a range of kernel_size_ratio):
pair_measurements = [
    ("model_a", "model_b",  115.2,  18.4),
    ("model_a", "model_c",   28.7,  17.1),
    # ... more pairs
]

gamma = fit_gamma_kernel_size(profiles, pair_measurements, hw_coefs)

plan = fractional_coresidency_v2(
    candidates,
    hw,
    interference_profiles=profiles,
    hw_coefs=hw_coefs,
    gamma_kernel_size=gamma,  # opt into V5; omit for v0.2 behaviour
)
```

On the reference bench's 7-model 19-pair corpus, V5 with operator-fitted γ reduces leave-one-pair-out (LOPO) cross-validation mean from 23.3% (v0.2 baseline) to 18.4%. Re-run LOPO on your corpus to decide whether to ship γ alongside your profiles.

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

## Algorithmic prior art

`nvinx`'s patterns are not novel algorithms — they are explicit, composable formulations of techniques the GPU multi-tenancy and DL-inference-serving literature has documented at a peer-reviewed level. The contribution of this package is to bring those techniques to the small-VRAM bioinformatics-bench audience, where the algorithms have not yet been packaged for direct use.

- **Pattern B (`fractional_coresidency`)** — algorithmic family established by *iGniter: Interference-Aware GPU Resource Provisioning for Predictable DNN Inference in the Cloud* (Xu et al., IEEE TPDS 2022, [10.1109/TPDS.2022.3232715](https://doi.org/10.1109/TPDS.2022.3232715)) and extended by *ECLIP: Energy-efficient and Practical Co-Location of ML Inference on Spatially Partitioned GPUs* (ISLPED 2025, [10.1109/ISLPED65674.2025.11261793](https://doi.org/10.1109/ISLPED65674.2025.11261793)). Both papers establish bin-packing + interference-aware placement for distinct ML inference models sharing a single GPU's VRAM. `nvinx`'s `fractional_coresidency` is the same algorithmic family applied to small-VRAM bioinformatics workloads (protein language models + classifiers + structure predictors) where the literature is empirically thinner.

- **Pattern A (`serial_handoff`)** — closely related to standard CPU-GPU pipeline overlap patterns documented across genomics-acceleration literature (e.g., GenPIP nanopore pipelining, SquiggleFilter virus detection accelerator). The single-bench framing is what differs; the algorithm itself is the canonical overlap pattern.

- **Pattern C (`ram_overflow`)** — directly invokes HuggingFace `accelerate`'s `device_map="auto"` mechanism. The contribution is the declarative interface; the offload mechanism is the well-documented `accelerate` capability for VRAM-exceeding model loading.

If your work cites `nvinx` for any of these patterns, please cite the underlying algorithmic-family reference as well — `nvinx` is the deployment-layer formulation, not the algorithmic primary source.

---

## Status and stability

v0.3.0a1 is **alpha** in-tree. v0.2.0a1 is the most recent PyPI release; v0.1.0 was the prior stable. Expect:

- Additions in v0.3.0a1: substrate-bound V5 kernel-size-ratio correction (`predict_pair_latency_queue_aware_v5` + `fit_gamma_kernel_size` companion); `predict_pair_latency` dispatcher and `fractional_coresidency_v2` both accept an optional `gamma_kernel_size` kwarg; calibration docs Step 6 extended with V5 fitting workflow; expanded test suite (42 passing).
- Additions in v0.2.0a1: substrate-native interference primitives (`nvinx.interference`), `fractional_coresidency_v2`, queue-aware prediction baseline.
- Backward compat: v0.2 and v0.1 API surfaces are unchanged. `fractional_coresidency_v2` with `gamma_kernel_size=None` (the default) reproduces v0.2 behaviour exactly; with `interference_profiles=None` it reproduces v0.1 behaviour exactly. V5 is opt-in via an operator-fitted γ on the operator's own substrate.
- Future: turnkey `nvinx.calibration` module (lift the operator-side calibration tooling into the public package); cross-substrate γ validation on at least one non-mobile substrate class; more patterns as edge cases surface.
- Pin your version. `SchedulingPlan` may add fields in minor releases.

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

pytest              # 42 tests (8 v0.1 patterns + 34 v0.2 / v0.3 interference)
ruff format --check .
ruff check .
```

42 tests in two files: `tests/test_patterns.py` (8; v0.1 patterns) and `tests/test_interference.py` (34; v0.2 queue-aware + v0.3 V5 + fit_gamma + dispatcher routing + backward-compat). CI runs on every push and PR across Python 3.10–3.12.

---

## License

MIT. See [LICENSE](LICENSE).

---

## Trademark disclaimer

*`nvinx` is not affiliated with NVIDIA Corporation or NGINX Inc. / F5, Inc. The name is a portmanteau reflecting the nginx-inspired upstream-routing metaphor applied to heterogeneous compute. If trademark concerns surface, the fallback package name is `nvginx`.*
