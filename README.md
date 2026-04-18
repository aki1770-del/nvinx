# nvinx

> nginx-style workload scheduler for limited-VRAM GPU bench setups.

**Status:** v0.1 pre-alpha. README pitch is a placeholder — full voice to be drafted in a dedicated session. Not yet published to PyPI.

*nvinx is not affiliated with NVIDIA Corporation or NGINX Inc. / F5, Inc. The name is a portmanteau reflecting the nginx-inspired upstream-routing metaphor applied to heterogeneous compute.*

## Patterns

nvinx packages three heterogeneous-compute scheduling patterns for researchers
running mixed model workloads on a single workstation with a small GPU:

- **`serial_handoff`** — when one GPU job saturates the card, CPU-only work
  runs in parallel in the same wall-clock window.
- **`fractional_coresidency`** — multiple small-VRAM models are packed into
  the shared VRAM budget via greedy bin-packing.
- **`ram_overflow`** — when a single model exceeds VRAM, layers are offloaded
  to system RAM via HuggingFace `accelerate`'s `device_map="auto"`.

See [`src/nvinx/patterns.py`](src/nvinx/patterns.py) for the API.

## Install (not yet published)

```bash
# once released to PyPI:
pip install nvinx

# from source:
git clone https://github.com/aki1770-del/nvinx
cd nvinx
pip install -e ".[dev]"
pytest
```

## License

MIT. See [LICENSE](LICENSE).
