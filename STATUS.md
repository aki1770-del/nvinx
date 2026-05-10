# Status — v0.2.0a1 in-tree (v0.1.0 released; v0.2 features added 2026-05-10)

**As of 2026-05-10:** v0.2.0a1 in-tree with backward-compat. v0.1.0 PyPI release remains intact (29 tests pass; no regression). v0.2 adds substrate-native queue-aware interference primitives derived from a multi-iteration empirical calibration on RTX A1000 4GB mobile substrate (calibration tooling stays in a separate research workspace, private).

## v0.2.0a1 additions

**New module `nvinx.interference`:**
- `HardwareCoefficients`, `InterferenceProfile`, `PairLookupEntry` dataclasses (frozen; operator-generated only per Option B-extended discipline)
- `predict_pair_latency_queue_aware()` — substrate-native formula: `latency_i = act_solo_i × (1 + θ_i × partner_act / (act_solo_i + partner_act)) + scheduling_delay`
- `max_kernel_rate_score()` — pre-filter heuristic (Spearman ρ=0.50 with measured slowdown)
- `asymmetry_predictor()` — act_solo_ratio for which-suffers-more (Spearman ρ=0.72 with asymmetry)
- `predict_pair_latency()` — tiered: lookup → queue-aware → fallback

**Enhanced `fractional_coresidency_v2()` in `nvinx.patterns`:**
- Same placement decision as v0.1 (greedy bin-pack); v0.2 adds diagnostic + advisory annotations
- Optional `interference_profiles`, `hw_coefs`, `pair_lookup`, `max_kernel_rate_threshold` parameters
- If profiles=None → equivalent to v0.1 `fractional_coresidency`

**21 new tests pass; 8 v0.1 tests unchanged.**

## v0.2 empirical foundation (the calibration journey)

The substrate-native queue-aware model emerged from iterative Sakichi 5-whys discipline:

| Iteration | Best LOPO mean error |
|---|---|
| iGniter linear with proxy | 142% |
| iGniter linear with real Nsight | 58% |
| iGniter sqrt formula | 37% |
| Cross-arch corpus exposed iGniter ceiling | 58% |
| **Queue-aware substrate-native (v0.3 D2 4-model)** | **15.8%** |
| 5-model corpus with same formula | 24.6% |

Key empirical findings:
- iGniter cache-pressure formula does NOT transfer to heterogeneous-kernel-rate substrates without real Nsight Compute
- Tetris cluster-variance (Liu lab IEEE TSC 2024) does NOT transfer to single-bench (no spatial dimension)
- Queue-aware per-model θ correlates continuously with kernel rate, with architecture-class variance
- Persistent ~30% LOPO outlier on 2-small-kernel pairs (formula limit; lookup safety net handles it)

## v0.2 scope and limits

- **Substrate-bound:** RTX A1000 mobile + Ampere sm_86; cross-substrate calibration required (v0.5 Track D)
- **Operator-controlled profiling:** `InterferenceProfile` + `HardwareCoefficients` are operator-generated; the package ships no published reference profile data (security discipline: published reference profiles would create a multi-tenant attack surface). Calibration tooling lives in a separate research workspace (private).
- **Lookup safety net:** for pairs where queue-aware confidence is low (e.g., 2-small-kernel), per-pair measured ground truth via `PairLookupEntry`
- **No production-grade calibration UX yet:** v0.2.0a1 is alpha; calibration is a research tool, not turnkey

## Validation summary (v0.1 — unchanged)

A 4 GB-class oncology research bench running multi-model heterogeneous inference (long-read sequencing basecaller → mutation classification + MHC-I binding prediction + structure prediction) was used as the named first customer for v0.1 validation. All three nvinx patterns mapped directly onto the bench's workload windows:

- **Pattern A** (`serial_handoff`) → long basecall windows (4–8 h overnight) + CPU-parallel work (variant calling, QC reporting, document scaffolding)
- **Pattern B** (`fractional_coresidency` / `_v2`) → small co-resident inference models (≤1 GB total in VRAM); mutation classifier + MHC-I binding predictor
- **Pattern C** (`ram_overflow`) → oversized structure prediction (>VRAM); layers spill to system RAM via HuggingFace `accelerate` at the documented 3–5× slowdown

The mapping surfaced one gap: clinical-vs-research pre-emption (where a new clinical case workload preempts a longer-running research training job) is not yet a named pattern. This is the candidate 4th pattern for v0.2.

## What this is and is not

- **Not a runtime.** nvinx returns placement decisions; you execute them with PyTorch, HuggingFace, vLLM, or your custom runtime.
- **Substrate-agnostic.** Works for any small-VRAM heterogeneous workload — bioinformatics inference, neoantigen ranking, protein structure prediction, custom multi-modal pipelines. Not coupled to a specific model class or runtime.
- **Not a competitor to `pollockjj/ComfyUI-MultiGPU` or `turboderp-org/exllamav3`.** Those are runtimes for specific workload classes (image generation, autoregressive LLMs). nvinx is scheduling logic — composable into either of those runtimes or into a custom one. If your workload is ComfyUI-shaped or pure-LLM, those incumbents serve you better than nvinx will.

## License

MIT. Trademark disclaimer in `README.md` (no NVIDIA / NGINX affiliation) remains. Fallback name `nvginx` reserved on PyPI.

## Roadmap

- **v0.1.0** — three patterns, the data model, configs, validated first customer (released 2026-04-27)
- **v0.2.0a1** — substrate-native queue-aware interference primitives + `fractional_coresidency_v2` (in-tree 2026-05-10; PyPI release pending)
- **v0.2/v0.3** — clinical-vs-research pre-emption (4th pattern), YAML config loader (the `Config.from_yaml()` referenced in `configs/bench_4gb.yaml`)
- **v0.4** — runtime adapters (PyTorch, HuggingFace, vLLM) as optional extras
- **v0.5+** — preprint preparation; cross-substrate validation; nonlinear formula refinement

## Contributing

The highest-value contribution is a case-study YAML describing your bench (hardware envelope, workload composition, which pattern fit, what changed). See `examples/case_studies/`. Bug reports with minimal reproduction also welcome. New pattern proposals: open an issue first so we can discuss scope.
