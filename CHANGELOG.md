# Changelog

All notable changes to `nvinx` are documented here. The project loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/spec/v2.0.0.html); pre-1.0 minor versions can break API.

---

## [0.3.0a2] — 2026-05-16 — substrate-class gate + γ-fit precision guidance (alpha)

### Added — Substrate-class detection (`nvinx.substrate`)

New module `nvinx.substrate` providing best-effort NVML-based runtime detection of the GPU substrate class (datacenter / mobile / unknown). Detection uses compute capability + SM count + total VRAM to classify; cached at module import time via `lru_cache`.

- `SubstrateInfo` dataclass — detected substrate parameters (name, SM count, total memory, compute capability, classification)
- `detect_substrate(device_index=0)` — returns `SubstrateInfo` (always; "unknown" if NVML or pynvml unavailable)
- `warn_if_datacenter_with_nonzero_gamma(gamma_kernel_size)` — advisory gate: emits `RuntimeWarning` when a non-zero `gamma_kernel_size` is passed AND the runtime substrate is detected as datacenter-class. **Does NOT override the operator's value** — operator-controlled discipline preserved per `docs/calibrating-your-substrate.md`.

### Changed — V5 substrate-class gate

`predict_pair_latency_queue_aware_v5` now calls `warn_if_datacenter_with_nonzero_gamma` when a non-zero γ is supplied. The warning alerts the operator that the v0.5 reference γ ≈ 0.7456 (fitted on a 4 GB Ampere mobile substrate) was empirically measured NOT to transfer to datacenter substrates: a first-party cross-substrate validation on an Ampere datacenter substrate (108-SM, 40 GB VRAM) in 2026-05-15 produced γ = 0.0331 (a 22.5× collapse), with V5 LOPO mean error 15.54 % vs V1 baseline 14.67 % (V5 makes datacenter predictions worse, not better). The warning recommends γ = 0 (reduces to V1 queue-aware baseline) on datacenter substrates, or operator γ-refitting via `nvinx.calibration`.

The gate is ADVISORY only. Operator override is respected. Suppress via:
```python
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="nvinx.substrate")
```

`fractional_coresidency_v2` inherits the gate transitively via its call to `predict_pair_latency` → `predict_pair_latency_queue_aware_v5`.

### Added — γ-fit precision warning (`fit_gamma_kernel_size`)

`fit_gamma_kernel_size` now emits `RuntimeWarning` when called with fewer than 6 pair measurements. A diagnostic deep-profile on the Ampere mobile native substrate in 2026-05-16 showed γ moved 10× between 10-pair (0.0331) and 3-pair (0.3164) fits on the same substrate — the γ-fit landscape is shallow at the 3-pair LOPO floor and the fitted value is corpus-sensitive. **Operator guidance: ≥6 pairs (≥5 models) for stable γ.**

### Honest scope

- The substrate-class gate is best-effort: detection requires NVML (`nvidia-ml-py` in the `[calibration]` extras OR system NVML library). If detection fails, no warning is emitted and the operator's `gamma_kernel_size` value is used as-is.
- The classification thresholds (SM count > 64 + VRAM ≥ 16 GB → datacenter; SM count ≤ 48 OR VRAM < 12 GB → mobile) are calibrated against known GPU lineups but may not perfectly classify intermediate substrates. Intermediate substrates classify as "unknown" and emit no warning.
- The "non-transfer to datacenter" finding is based on one first-party measurement (Ampere datacenter 108-SM 40 GB) plus the published iGniter cross-substrate evidence on a Volta-class datacenter GPU. Additional substrate-class measurements (Volta first-party, Hopper, mid-tier mobile families) are planned future work.

### Backward compatibility

- v0.3.0a1 API surface is unchanged. Existing callers of `predict_pair_latency_queue_aware_v5`, `predict_pair_latency`, `fractional_coresidency_v2`, `fit_gamma_kernel_size` see no behaviour change other than the new advisory `RuntimeWarning`s. The warnings can be filtered.
- v0.2 + v0.1 API surfaces unchanged.

### Tests

- 21 new tests in `tests/test_substrate.py`: classifier unit tests against known GPU lineups across Ampere datacenter / Hopper datacenter / Volta datacenter / Ada datacenter / Ampere mobile / Ada mobile / intermediate-edge / no-data / partial-data; `detect_substrate` integration test; `warn_if_datacenter_with_nonzero_gamma` behaviour tests (None/0.0 skip; datacenter fires; mobile does not fire; unknown does not fire); `fit_gamma_kernel_size` <6-pairs warning test.
- 70 tests total: 49 existing + 21 new. All green.

---

## [0.3.0a1] — 2026-05-12 — substrate-bound V5 kernel-size-ratio correction + turnkey calibration (alpha)

### Added — V5 substrate-bound kernel-size-ratio correction (`nvinx.interference`)

- `predict_pair_latency_queue_aware_v5(profile_a, profile_b, hw, *, gamma_kernel_size=None)` — extends the v0.2 queue-aware formula with one optional empirical scalar `gamma_kernel_size` that captures the asymmetric blocking of small-kernel models by large-kernel partners. Formula: `latency_self = act_self × (1 + theta × partner_frac × (1 + gamma × kernel_size_ratio)) + scheduling_delay + baseidle_self`. When `gamma_kernel_size` is `None` (default) or `0.0`, the function is bit-identical to v0.2's `predict_pair_latency_queue_aware`.
- `fit_gamma_kernel_size(profiles, pair_measurements, hw)` — closed-form weighted least-squares fit for the substrate-specific γ given operator-fitted thetas. Minimises *relative* residuals (`((pred − observed) / observed)²`) via pure-stdlib weighted-LS — no `numpy` / `scipy` dependency added. Matches the residual-norm convention an operator gets from running `scipy.optimize.least_squares` on relative residuals (the appropriate norm for substrates with wide `act_solo` spread across the corpus).
- `predict_pair_latency` dispatcher accepts a new optional `gamma_kernel_size` kwarg; routes to the V5 path when supplied (source label `"queue_aware_v5"`) and to the v0.2 path otherwise (source label `"queue_aware"` unchanged).
- `fractional_coresidency_v2` accepts a new optional `gamma_kernel_size` kwarg; threads through to per-pair predictions via the dispatcher. When `None` (default), v0.2 behaviour is reproduced exactly.
- New "Step 6: Fit `gamma_kernel_size` (optional v0.6 alpha)" in [`docs/calibrating-your-substrate.md`](docs/calibrating-your-substrate.md): kernel-size-ratio mechanism; when V5 helps (asymmetric small-kernel-vs-large-kernel pairs on small-SM substrates); when V5 may not help (datacenter-class GPUs with many SMs; cache-contention-dominated physics; iGniter cross-substrate evidence); operator workflow for fitting γ via `fit_gamma_kernel_size`.
- 13 new tests in `tests/test_interference.py` covering: backward-compat (γ=None/0.0 produce bit-identical output to v0.2), V5 inflation on asymmetric kernel-size-ratio pairs, theta-required and kernel-zero safety, synthetic γ recovery, error-path coverage, dispatcher routing source labels, `fractional_coresidency_v2` accepts new kwarg without breaking existing callers.

### Added — Turnkey calibration submodule (`nvinx.calibration`)

- New subpackage `nvinx.calibration` lifting the per-step calibration tooling into a library API + CLI so operators no longer need to write their own scripts. Public surface:
  - `CalibrationResult` dataclass — end-to-end output (`HardwareCoefficients` + fitted `InterferenceProfile`s + pair measurements + LOPO summary + optional V5 γ).
  - `ProfileTarget` dataclass — operator-supplied bundle of `name` + `inference_fn` + `sample_input` + `loader_code` (used by the ncu subprocess) + `canonical_batch`.
  - `sweep_hardware()` — substrate-agnostic idlef polynomial + powerp linear sweep. Detects GPU name / nominal clock / TDP via `nvidia-smi`.
  - `profile_model(loader, ...)` — per-model standalone profiling with VRAM-tight factory pattern (loader called inside, parent's model freed before ncu subprocess, re-loaded for k_l2 measurement).
  - `validate_pair(target_a, target_b, ...)` — cross-pair co-located latency measurement on separate CUDA streams.
  - `fit_thetas(profiles, pair_measurements, hw)` — V1 baseline theta fit via `scipy.optimize.least_squares` on relative residuals.
  - `fit_v5(profiles, pair_measurements, hw)` — joint theta + V5 γ fit (bit-equivalent to the v0.5 H5 reference).
  - `lopo_cross_validate(profiles, pair_measurements, hw, *, refit_v5=False)` — leave-one-pair-out cross-validation summary.
  - `apply_thetas(profiles, thetas)` — construct a new dict of `InterferenceProfile` with fitted theta values applied.
  - `run_calibration(model_loaders, *, output_dir, fit_v5_gamma=False, ...)` — end-to-end orchestrator.
- New CLI entry point `nvinx-calibrate` (`[project.scripts]`). Operator supplies `--model module.path:loader_fn` specs (≥ 3 required); CLI imports each loader via `importlib`, runs the full pipeline, writes `<output_dir>/calibration_result.json`.
- New optional dependencies via extras: `pip install 'nvinx[calibration]'` installs `numpy >= 1.24`, `scipy >= 1.10`, `nvidia-ml-py >= 12.0`. The base nvinx install remains lightweight (`pyyaml` only). System prerequisite: `ncu` (Nsight Compute) installed and accessible (either `sudo ncu` or `NVreg_RestrictProfilingToAdminUsers=0`).
- 7 new tests in `tests/test_calibration_fit.py` (synthetic-data; pure math; no GPU required). All gated on `pytest.importorskip("scipy")` so they cleanly skip on default installs without the `[calibration]` extras.
- Updated [`docs/calibrating-your-substrate.md`](docs/calibrating-your-substrate.md) with a new "Turnkey calibration via `nvinx.calibration`" section documenting the library API + CLI examples + per-step API reference.

### Honest scope

- The reference value `γ ≈ 0.75` was fitted on an extended 7-model 19-pair corpus on a 4 GB RTX A1000 mobile substrate with a transformer/LLM workload mix. Earlier iterations on the same substrate: 6-model 15-pair → γ ≈ 0.44; 4-model 6-pair → γ → 0. Operators on other substrates **MUST** refit via `fit_gamma_kernel_size`.
- V5 may not help — or may hurt — on datacenter-class GPUs with many SMs (Volta / Ampere / Hopper datacenter; 80+ SMs) where L2 cache contention is the dominant interference physics. The published `iGniter` evaluation on a Volta-class datacenter GPU + CNN saw max ~1.35× slowdown at 5 co-located workloads; the reference bench saw max ~12.11× slowdown at 2 workloads. The ~30× magnitude gap is the empirical signature of fundamentally different interference physics across substrate classes.
- Cross-substrate validation of V5 on a non-mobile substrate is not yet completed; the formula is alpha pending that evidence.

### Backward compatibility

- v0.2 API surface is unchanged. Existing callers of `predict_pair_latency_queue_aware`, `predict_pair_latency` (without `gamma_kernel_size`), `fractional_coresidency_v2` (without `gamma_kernel_size`) see no behaviour change.
- v0.1 API surface is unchanged.

### Tests

- 49 tests in total: 42 always-available (8 v0.1 patterns + 34 v0.2 / v0.3 interference) + 7 calibration-extras-gated. The 7 calibration tests gracefully skip on default installs without `[calibration]` extras. CI green on Python 3.10 / 3.11 / 3.12.

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
