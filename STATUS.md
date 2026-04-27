# Status — v0.1 release candidate

**As of 2026-04-27:** scaffold complete (3 patterns, 8 passing tests, README, CI, configs, examples), validated against a real 4 GB-class oncology research bench workload (see `examples/case_studies/oncology_4gb_bench.py`). v0.1 is the first public release.

## Validation summary

A 4 GB-class oncology research bench running multi-model heterogeneous inference (long-read sequencing basecaller → mutation classification + MHC-I binding prediction + structure prediction) was used as the named first customer for v0.1 validation. All three nvinx patterns mapped directly onto the bench's workload windows:

- **Pattern A** (`serial_handoff`) → long basecall windows (4–8 h overnight) + CPU-parallel work (variant calling, QC reporting, document scaffolding)
- **Pattern B** (`fractional_coresidency`) → small co-resident inference models (≤1 GB total in VRAM); mutation classifier + MHC-I binding predictor
- **Pattern C** (`ram_overflow`) → oversized structure prediction (>VRAM); layers spill to system RAM via HuggingFace `accelerate` at the documented 3–5× slowdown

The mapping surfaced one gap: clinical-vs-research pre-emption (where a new clinical case workload preempts a longer-running research training job) is not yet a named pattern. This is the candidate 4th pattern for v0.2.

## What this is and is not

- **Not a runtime.** nvinx returns placement decisions; you execute them with PyTorch, HuggingFace, vLLM, or your custom runtime.
- **Substrate-agnostic.** Works for any small-VRAM heterogeneous workload — bioinformatics inference, neoantigen ranking, protein structure prediction, custom multi-modal pipelines. Not coupled to a specific model class or runtime.
- **Not a competitor to `pollockjj/ComfyUI-MultiGPU` or `turboderp-org/exllamav3`.** Those are runtimes for specific workload classes (image generation, autoregressive LLMs). nvinx is scheduling logic — composable into either of those runtimes or into a custom one. If your workload is ComfyUI-shaped or pure-LLM, those incumbents serve you better than nvinx will.

## License

MIT. Trademark disclaimer in `README.md` (no NVIDIA / NGINX affiliation) remains. Fallback name `nvginx` reserved on PyPI.

## Roadmap

- **v0.1** — three patterns, the data model, configs, validated first customer (this release)
- **v0.2** — clinical-vs-research pre-emption (4th pattern), YAML config loader (the `Config.from_yaml()` referenced in `configs/bench_4gb.yaml`)
- **v0.3** — runtime adapters (PyTorch, HuggingFace, vLLM) as optional extras

## Contributing

The highest-value contribution is a case-study YAML describing your bench (hardware envelope, workload composition, which pattern fit, what changed). See `examples/case_studies/`. Bug reports with minimal reproduction also welcome. New pattern proposals: open an issue first so we can discuss scope.
