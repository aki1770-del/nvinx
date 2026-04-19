# configs/

Sample `HardwareSpec + list[ModelSpec]` YAML files for three reference benches.

## Status

**Forward-looking design artifacts.** The v0.2 release will ship a
`Config.from_yaml()` loader that parses these files directly into the catalog
dataclasses. For now, the YAMLs serve as design previews — hand-translate to
Python when building a `SchedulingPlan` (see `examples/04_composed_day.py`).

## Files

| File | Bench | VRAM | RAM | Representative GPU |
|---|---|---|---|---|
| [`bench_4gb.yaml`](bench_4gb.yaml) | small | 4 GB | 32 GB | RTX A1000, GTX 1650 |
| [`bench_8gb.yaml`](bench_8gb.yaml) | mid | 8 GB | 64 GB | RTX A2000, RTX 3060 Ti |
| [`bench_24gb.yaml`](bench_24gb.yaml) | consumer-large | 24 GB | 128 GB | RTX 3090, RTX 4090 |

## Schema (preview)

```yaml
hardware:
  vram_gb: <float>
  ram_gb: <float>
  cpu_cores: <int>

models:
  - name: <str>
    vram_gb: <float>
    residency: GPU_EXCLUSIVE | GPU_SHARED | CPU_ONLY | GPU_RAM_OVERFLOW
    cpu_fallback_supported: <bool, default false>
    ram_overflow_supported: <bool, default false>
    ram_gb_needed: <float | null, default null>
```

Field names match `nvinx.catalog.HardwareSpec` and `nvinx.catalog.ModelSpec`
exactly; the v0.2 loader will be a near-trivial unmarshal.
