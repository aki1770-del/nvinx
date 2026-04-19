# Case studies

This directory collects real-world case studies that exercise the three nvinx
patterns on actual hardware and workloads. Contributions welcome — they're the
highest-signal feedback we get.

## Submission format

Open a pull request with a single YAML file: `<your-handle>_<bench-name>.yaml`.

```yaml
submitter: your-github-handle
date: 2026-04-19
hardware:
  vram_gb: 4.0
  ram_gb: 32.0
  cpu_cores: 8
  model: RTX A1000          # optional, informational only

workload:
  domain: bioinformatics    # or: RAG, CV, NLP, structural-biology, other
  description: >
    One-paragraph description of what you're trying to do and why small-GPU
    scheduling matters. Workload names can stay vague if sensitive.

  models:
    - name: long_inference
      vram_gb: 3.5
      residency: GPU_EXCLUSIVE
    - name: feature_extractor
      vram_gb: 0.0
      residency: CPU_ONLY

experience:
  pattern_tried: serial_handoff      # or fractional_coresidency, ram_overflow
  old_wall_clock_hours: 14
  new_wall_clock_hours: 8
  what_broke: null                   # or: description of edge cases, errors
  suggested_improvements: null
```

## What counts as a useful case study

- Real hardware, real workloads. Toy examples belong in `examples/`.
- A before/after wall-clock if possible — even a rough estimate.
- Honest failures. If a pattern didn't help, that's the most valuable signal.
- Feel free to keep workload names vague if they're proprietary. We want the
  hardware and scheduling pattern data, not your IP.

## Anonymization

No anonymization requirement. But if you want to scrub identifying details,
generic substitutions are fine:

- `long_inference` instead of named model
- `feature_extractor` / `classifier_a` / `embedder` instead of project-specific names
- Domain keyword only (e.g. `bioinformatics`, `RAG`) instead of project name

## Licensing

Case-study submissions are MIT-licensed along with the rest of the repo. By
submitting a PR you agree to that license.
