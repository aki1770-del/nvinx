# Changelog

All notable changes to `nvinx` are documented here. The project loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html); pre-1.0 minor versions can break API.

---

## [0.3.0a1] — 2026-05-12 — substrate-bound V5 kernel-size-ratio correction (alpha)

### Added

- `predict_pair_latency_queue_aware_v5(profile_a, profile_b, hw, *, gamma_kernel_size=None)` — extends the v0.2 queue-aware formula with one optional empirical scalar `gamma_kernel_size` that captures the asymmetric blocking of small-kernel models by large-kernel partners. Formula: `latency_self = act_self × (1 + theta × partner_frac × (1 + gamma × kernel_size_ratio)) + scheduling_delay + baseidle_self`. When `gamma_kernel_size` is `None` (default) or `0.0`, the function is bit-identical to v0.2's `predict_pair_latency_queue_aware`.
- `fit_gamma_kernel_size(profiles, pair_measurements, hw)` — closed-form weighted least-squares fit for the substrate-specific γ given operator-fitted thetas. Minimises *relative* residuals (`((pred − observed) / observed)²`) via pure-stdlib weighted-LS — no `numpy` / `scipy` dependency added. Matches the residual-norm convention an operator gets from running `scipy.optimize.least_squares` on relative residuals (the appropriate norm for substrates with wide `act_solo` spread across the corpus).
- `predict_pair_latency` dispatcher accepts a new optional `gamma_kernel_size` kwarg; routes to the V5 path when supplied (source label `"queue_aware_v5"`) and to the v0.2 path otherwise (source label `"queue_aware"` unchanged).
- `fractional_coresidency_v2` accepts a new optional `gamma_kernel_size` kwarg; threads through to per-pair predictions via the dispatcher. When `None` (default), v0.2 behaviour is reproduced exactly.
- New "Step 6: Fit `gamma_kernel_size` (optional v0.6 alpha)" in [`docs/calibrating-your-substrate.md`](docs/calibrating-your-substrate.md): kernel-size-ratio mechanism; when V5 helps (asymmetric small-kernel-vs-large-kernel pairs on small-SM substrates); when V5 may not help (datacenter-class GPUs with many SMs; cache-contention-dominated physics; iGniter cross-substrate evidence); operator workflow for fitting γ via `fit_gamma_kernel_size`.
- 13 new tests in `tests/test_interference.py` covering: backward-compat (γ=None/0.0 produce bit-identical output to v0.2), V5 inflation on asymmetric kernel-size-ratio pairs, theta-required and kernel-zero safety, synthetic γ recovery, error-path coverage, dispatcher routing source labels, `fractional_coresidency_v2` accepts new kwarg without breaking existing callers.

### Honest scope

- The reference value `γ ≈ 0.75` was fitted on an extended 7-model 19-pair corpus on a 4 GB RTX A1000 mobile substrate with a transformer/LLM workload mix. Earlier iterations on the same substrate: 6-model 15-pair → γ ≈ 0.44; 4-model 6-pair → γ → 0. Operators on other substrates **MUST** refit via `fit_gamma_kernel_size`.
- V5 may not help — or may hurt — on datacenter-class GPUs with many SMs (Volta / Ampere / Hopper datacenter; 80+ SMs) where L2 cache contention is the dominant interference physics. The published `iGniter` evaluation on a Volta-class datacenter GPU + CNN saw max ~1.35× slowdown at 5 co-located workloads; the reference bench saw max ~12.11× slowdown at 2 workloads. The ~30× magnitude gap is the empirical signature of fundamentally different interference physics across substrate classes.
- Cross-substrate validation of V5 on a non-mobile substrate is not yet completed; the formula is alpha pending that evidence.

### Backward compatibility

- v0.2 API surface is unchanged. Existing callers of `predict_pair_latency_queue_aware`, `predict_pair_latency` (without `gamma_kernel_size`), `fractional_coresidency_v2` (without `gamma_kernel_size`) see no behaviour change.
- v0.1 API surface is unchanged.

### Tests

- 42 tests pass (8 v0.1 patterns + 34 v0.2 / v0.3 interference). CI green on Python 3.10 / 3.11 / 3.12.

---

## [0.2.0a1] — 2026-05-10 — substrate-native queue-aware interference primitives (alpha)

### Added

- New module `nvinx.interference` with `HardwareCoefficients`, `InterferenceProfile`, `PairLookupEntry` dataclasses (all frozen; operator-generated only — the package ships no published reference profile data).
- `predict_pair_latency_queue_aware(profile_a, profile_b, hw)` — substrate-native queue-aware interference prediction.
- `max_kernel_rate_score(profiles)` — pre-filter heuristic (Spearman ρ ≈ 0.50 with measured slowdown on the validation bench).
- `asymmetry_predictor(profile_a, profile_b)` — `act_solo` ratio predicting which model suffers more under co-residency (ρ ≈ 0.72).
- `lookup_pair_latency(pair, lookup)` / `PairLookupEntry` — per-pair measured ground-truth safety net for known high-error pairs.
- `predict_pair_latency(profile_a, profile_b, hw, *, pair_lookup=None)` — tiered dispatcher: lookup → queue-aware → fallback.
- `fractional_coresidency_v2(..., *, interference_profiles, hw_coefs, pair_lookup, max_kernel_rate_threshold)` — v0.1 placement decision unchanged; v0.2 augments the plan's `notes` with interference predictions when profiles are provided.
- `docs/calibrating-your-substrate.md` — full operator workflow for fitting `HardwareCoefficients` and per-model `InterferenceProfile` on the operator's substrate.

### Honest scope

- The queue-aware formula was validated on one substrate (4 GB RTX A1000 mobile + heterogeneous transformer/LLM workload mix). LOPO mean error ~16% on the 4-model 6-pair corpus; ~25% on the 5-model 10-pair extended corpus. Persistent ~30% LOPO outlier on 2-small-kernel pairs (formula limit; `PairLookupEntry` safety net handles those).

### Backward compatibility

- v0.1 API surface unchanged. `fractional_coresidency_v2` with `interference_profiles=None` reproduces v0.1 `fractional_coresidency` behaviour exactly.

### Tests

- 29 tests (8 v0.1 patterns + 21 v0.2 interference). CI green on Python 3.10 / 3.11 / 3.12.

---

## [0.1.0] — 2026-04-27 — initial PyPI release

### Added

- Three placement patterns for limited-VRAM GPU bench setups:
  - **Pattern A — `serial_handoff`**: CPU work during a GPU-exclusive window.
  - **Pattern B — `fractional_coresidency`**: pack small-VRAM models into a shared budget (greedy bin-pack).
  - **Pattern C — `ram_overflow`**: hint construction for HuggingFace `accelerate` device-map auto-offload.
- Core data model in `nvinx.catalog`: `HardwareSpec`, `ModelSpec`, `Residency` enum, `SchedulingPlan`.
- 8 tests covering all three patterns.

### Notes

- Anchored to peer-reviewed algorithmic-family references (iGniter at IEEE TPDS 2022/2023; ECLIP at ISLPED 2025; see README's "Algorithmic prior art" section).
- MIT license.
