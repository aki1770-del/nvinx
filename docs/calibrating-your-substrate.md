# Calibrating Your Substrate for v0.2 Interference Primitives

This guide walks an operator through generating `HardwareCoefficients` and per-model `InterferenceProfile` objects for their own bench, so the v0.2 `fractional_coresidency_v2` interference predictions reflect *their* substrate.

**Why this matters.** The queue-aware formula in `nvinx.interference` is *empirical*. Its coefficients were fitted on one specific bench (4 GB RTX A1000 mobile + heterogeneous transformer/LLM workload). If you use those published coefficients on a different GPU, model class, or driver, the predictions will be wrong. This is not a defect of the formula — it is the substrate-binding nature of empirical interference modeling. Recalibrate for your bench.

**This guide covers** what to measure, in what order, and which tools to use. It does not provide a turnkey calibration script — the calibration tooling currently lives separately as a research artifact (see "Reference implementation" below). v0.2 is alpha; turnkey is a v0.3+ goal.

---

## Prerequisites

- An NVIDIA GPU with sufficient VRAM for at least one model + workspace overhead.
- NVIDIA driver + CUDA matching your PyTorch build (`torch.cuda.is_available()` returns `True`).
- **Nsight Compute (`ncu`)** installed and accessible. On Ubuntu: `sudo apt install nsight-compute` (multiverse repo). Older `nvprof` legacy tool is usually too imprecise for the L2 metrics needed.
- **NVIDIA driver perf-counter access** for non-root users — by default, `ncu` returns `ERR_NVGPUCTRPERM`. Workarounds: (a) run `ncu` via `sudo`, or (b) load the driver with `NVreg_RestrictProfilingToAdminUsers=0`. Option (a) is simpler for one-off calibration.
- **MPS daemon optional** if you want spatial GPU partitioning (`nvidia-cuda-mps-control`). v0.2 calibration does not require MPS.
- Python ≥ 3.10 with the model framework you use (PyTorch + HuggingFace transformers is most common).

---

## What to measure

For your substrate, you need:

### `HardwareCoefficients` (one-time per bench)

| Field | What it is | How to measure |
|---|---|---|
| `nominal_freq_mhz` | GPU base clock when not power-capped | `nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits` |
| `tdp_watts` | GPU thermal design power (the cap above which frequency drops) | `nvidia-smi --query-gpu=power.max_limit --format=csv,noheader,nounits` |
| `idlef_polynomial` | Per-inference scheduling delay added at N concurrent workloads | Fit polynomial to delays measured at N = 1, 2, 3, 4, 5 concurrent micro-benchmark kernels |
| `powerp_linear` | Frequency reduction slope when power exceeds TDP | Fit slope of `frequency` vs `power.draw` samples during a sustained heavy workload |
| `substrate_name` | A human-readable label for the bench (e.g., `"rtx_a1000_4gb"`) | string |

### `InterferenceProfile` (one per model)

| Field | What it is | How to measure |
|---|---|---|
| `name` | Identifier you'll use in placement | string |
| `kernels` | Number of CUDA kernels per inference | `torch.profiler.profile(activities=[...CUDA])` and count distinct CUDA events |
| `baseidle_ms` | Standalone idle/scheduling time (query latency minus GPU active time) | Difference between wall-clock latency and CUDA event-timed active time, both measured on a warmed standalone run |
| `act_solo_ms` | Standalone GPU active time per inference | CUDA event start-to-end timing on a warmed standalone run; median over ≥5 trials |
| `l2_saturation_pct` | L2 cache sector throughput as % of peak sustained | `ncu --metrics lts__t_sectors.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum` then duration-weighted average across kernels |
| `theta` | Queue-aware sensitivity coefficient | Fitted by least-squares from cross-pair co-located measurements (see fitting step below) |
| `power_w` | Sustained power draw during inference | `nvidia-smi --query-gpu=power.draw` median over a multi-second sustained run |
| `architecture_class` | Tag like `"encoder_transformer"`, `"decoder_transformer"`, `"encoder_decoder"`, `"cnn"`, etc. | string; advisory only |

The `theta` field is the substrate-native parameter; it cannot be measured directly. It is *fitted* from the slowdowns you observe when models co-reside.

### Cross-pair measurements (for fitting `theta`)

For each pair of models (A, B) you want to admit together, measure the **co-located** latency — A and B running concurrently on separate CUDA streams — and the standalone latencies. The slowdown ratio per model under co-residency is the input to the `theta` fit.

A minimum-viable fit needs **3 distinct models** (so you have C(3,2) = 3 pairs = 6 measurements for 3 thetas to fit). More models = more pairs = more robust fit. The reference bench used 4–5 models depending on the iteration.

---

## Calibration order

1. **Hardware sweep first** (one-time per bench). Idlef polynomial + powerp linear. ~5 minutes. Doesn't depend on any model.

2. **Per-model standalone profile** for each model in your candidate set. ~5–10 minutes per model under `ncu` (the L2 measurement is the slow part — `ncu` profiles every kernel individually).

3. **Cross-pair co-located measurement** for each pair you want fitted. ~30–60 seconds per pair (multiple trials).

4. **Fit `theta_i` per model** by least-squares minimizing the residual between predicted and measured co-located latencies. With N models and C(N,2) pairs, this is a small unconstrained nonlinear regression — `scipy.optimize.least_squares` with `bounds=(0, inf)` works.

5. **Validate** by leave-one-pair-out cross-validation. If LOPO mean error is acceptable for your SLO budget, ship the profiles. If not, either:
   - Add more models and re-fit (more data → more stable thetas)
   - Build a per-pair lookup table (`PairLookupEntry`) for the high-error pairs as a safety net
   - Accept the formula as advisory only; rely on lookup primarily
   - Proceed to step 6 (V5 γ fit) if your LOPO outliers cluster on kernel-size-asymmetric pairs

6. **Optional: fit V5 `gamma_kernel_size` (v0.6 alpha; substrate-bound).** See the next section.

---

## Step 6: Fit `gamma_kernel_size` (optional v0.6 alpha)

The v0.2 queue-aware formula treats each side of a co-resident pair symmetrically in the partner-time-fraction term. On substrates where the dominant interference mechanism is the GPU scheduler queue rather than L2 cache pressure, an iterative calibration journey on a 4 GB RTX A1000 mobile substrate found that a single additional empirical term reduces LOPO mean by ~5 pp on transformer/LLM workload mixes. The reference value `γ ≈ 0.75` cited in this guide and in the `predict_pair_latency_queue_aware_v5` docstring was fitted on an **extended 7-model 19-pair corpus** (the 4-model corpus from the v0.2 baseline plus three additional decoder/encoder models that surfaced the kernel-size-asymmetry pattern); an earlier 6-model 15-pair fit on the same substrate yielded a smaller γ (≈ 0.44), and a 4-model 6-pair fit yielded γ → 0 (the asymmetry signal does not appear with too few or too uniform models in the corpus). LOPO mean improvement on the 7-model corpus: 23.3% baseline → 18.4% with V5. The formula:

```
partner_frac      = act_partner / (act_self + act_partner)
kernel_size_ratio = (act_partner / kernels_partner) / (act_self / kernels_self)
latency_self      = act_self × (1 + theta_self × partner_frac
                                  × (1 + gamma × kernel_size_ratio))
                  + scheduling_delay
```

What `kernel_size_ratio` captures: when a small-kernel-self model is paired with a large-kernel partner, every small-self kernel may queue behind a long-running large-partner kernel. The asymmetric blocking is not captured by `partner_frac` alone — it depends on the *duration ratio* of each side's individual kernels. The correction is a single scalar `gamma_kernel_size` empirical per substrate.

### When V5 helps

- Heterogeneous kernel-size mix in your corpus (e.g., a decoder LLM with many small kernels co-located with an encoder model with fewer larger kernels).
- Small-SM substrates where queue contention dominates (the reference bench has 16 SMs; the SM pool saturates quickly).
- Your v0.2 LOPO outliers cluster on pairs with high `kernel_size_ratio` (you can check via the `kernel_duration_ms` property on each profile).

### When V5 may NOT help (or may hurt)

- Datacenter-class GPUs with many SMs (V100/A100/H100 — 80+ SMs) where the SM pool absorbs multiple co-located workloads and L2 cache contention is the dominant interference physics. The published `iGniter` evaluation on V100 + CNN saw max ~1.35× slowdown at 5 co-located workloads (cache-contention-dominated); the reference bench saw max ~12.11× slowdown at 2 workloads (queue-contention-dominated). The ~30× magnitude gap is the empirical signature of fundamentally different interference physics — V5's γ ≈ 0.75 reference fit will be wrong by an unknown sign on the iGniter substrate class.
- Same-architecture corpora (V5 collapses to v0.2 behaviour when all pairs have `kernel_size_ratio ≈ 1`).
- Workloads where measurement noise on `k_l2` is the dominant source of LOPO error (per the v0.5 H3 finding, multi-trial averaging was rejected as null-result on the reference substrate, but a different substrate may have different noise characteristics).

**Use v0.2 if** you have not collected calibration data on your substrate; the published V5 γ value will not transfer.

### How to fit

```python
from nvinx import (
    HardwareCoefficients,
    InterferenceProfile,
    fit_gamma_kernel_size,
)

# Step 1-5 already produced these (theta fitted on each profile):
hw_coefs: HardwareCoefficients = ...
profiles: dict[str, InterferenceProfile] = ...

# Cross-pair co-located measurements (one row per pair):
pair_measurements = [
    ("esm2_long",  "qwen_05b",     115.2,  18.4),
    ("esm2_short", "qwen_05b",      28.7,  17.1),
    ("esm2_long",  "whisper_base", 119.8,  62.3),
    # ... ideally ≥ 6 pairs spanning a range of kernel_size_ratio values
]

gamma = fit_gamma_kernel_size(profiles, pair_measurements, hw_coefs)
# Now `gamma` is your substrate-specific V5 coefficient.
```

### How to use

```python
from nvinx import fractional_coresidency_v2

plan = fractional_coresidency_v2(
    candidates,
    hw,
    interference_profiles=profiles,
    hw_coefs=hw_coefs,
    gamma_kernel_size=gamma,  # opt into V5 routing
)
```

`fractional_coresidency_v2` with `gamma_kernel_size=None` (the default) reproduces v0.2 queue-aware behaviour exactly. Existing v0.2 callers are unaffected.

### Validate the V5 fit

Re-run leave-one-pair-out CV using `predict_pair_latency_queue_aware_v5` and compare against your v0.2 LOPO. If V5 LOPO is lower on your corpus, ship γ alongside the profiles. If V5 LOPO is comparable or worse, stay on v0.2 — the kernel-size-ratio mechanism is not the dominant source of error on your substrate.

A single-parameter fit on too few pairs overfits; aim for ≥ 6 pairs spanning a range of `kernel_size_ratio` (mix small-kernel and large-kernel models in your corpus).

---

## Reference implementation

The calibration tooling that produced the reference results lives in a separate research workspace (private) and is research-shape — not part of the public `nvinx` package. Its layout, for orientation when you build your own:

```
research/v0_2_calibration/
  ├── coefficients.py            # dataclass definitions (compatible with nvinx.interference)
  ├── igniter_predict.py         # legacy iGniter formula (kept for comparison)
  ├── ncu_l2_profiler.py         # ncu wrapper for L2 saturation measurement
  ├── profile_hardware.py        # idlef + powerp sweeps
  ├── profile_model.py           # per-model coefficient fitting
  ├── queue_aware_model.py       # theta fitting via least_squares + LOPO CV
  ├── validate_pair.py           # concurrent stream pair latency measurement
  └── run_calibration.py         # end-to-end orchestrator
```

Operators who want to calibrate their own bench should write their own calibration scripts following this shape. The `nvinx.interference` dataclass field tables in this guide tell you what each script needs to produce.

A v0.3+ goal is to lift a stable subset of that tooling into `nvinx.calibration` as a turnkey operator-facing module. Until then, calibration is operator-driven.

---

## Honest scope of v0.2 calibration

- **One substrate validated.** The reference findings come from a single 4 GB RTX A1000 mobile GPU. Cross-substrate generalization is unverified. A 16 GB consumer GPU may need different idlef polynomial. An A100 may need different powerp behaviour. Your calibration is yours.

- **Architecture-class variance.** The `theta` coefficient correlates roughly with kernel rate (small-fast-kernel models suffer more queue contention than big-slow-kernel models), but the correlation is not deterministic across architecture classes — a decoder-transformer LLM had lower `theta` than an encoder-transformer at higher kernel rate, on the reference bench. If your corpus has diverse architectures, expect some scatter.

- **Persistent ~30% LOPO outlier on 2-small-kernel pairs.** When both models in a pair have high `theta` (both small-kernel), the simple per-model `theta` formula under-predicts the slowdown. The `PairLookupEntry` safety net handles this case for known pairs.

- **Run-to-run measurement noise.** The `theta` self-co-location measurement is sensitive to noise. Multi-trial averaging is recommended. Coefficient-of-variation on `theta` across runs can exceed 30% on noisy benches.

- **`act_solo_ms` measured under your driver state.** If you disable cuDNN for ncu compatibility (some conv-heavy models like Whisper require this), do so consistently across all measurements — pair latencies measured with cuDNN enabled and standalone baselines measured with it disabled give garbage slowdowns.

These are not blockers — they are the empirical-modeling reality. Calibrate honestly, document your bench, and your own users will benefit from the discipline.

---

## When to use v0.1 vs v0.2

- **Use v0.1 `fractional_coresidency`** if you're prototyping or your placement decisions are obvious from VRAM alone. It's the right answer when the question is "do they fit?"

- **Use v0.2 `fractional_coresidency_v2`** when you need to know *whether the placement will hit SLO* — and you've calibrated for your substrate. It's the right answer when the question is "will they interfere?"

- **Use neither** if your substrate has just one model that fully saturates the GPU (`Pattern A` is your tool), or if your only model exceeds VRAM (`Pattern C`).

---

## Pointers

- The IEEE TPDS papers cited in the algorithmic prior art section of the README are the foundational external references; this guide assumes you've read at least the iGniter abstract.
- The empirical journey that produced the v0.2 formula is documented in private research-workspace closure seals; their published essence is the `Status` and `Honest scope` sections of the nvinx STATUS.md and this guide.

If you calibrate your bench and want to share, a `examples/case_studies/<bench-name>.yaml` PR is the highest-value contribution. Document hardware envelope, models profiled, fitted thetas, LOPO error — anonymize as needed.
