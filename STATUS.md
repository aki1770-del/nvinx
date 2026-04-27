# Status — parked at v0.1 (not released)

**As of 2026-04-26:** scaffold-complete (3 patterns, 8 passing tests, README, CI, configs, examples) but **not released**. PyPI `nvinx 0.0.1` is a placeholder squat for name reservation only — it is not v0.1 and should not be installed.

## Why parked

A landscape audit at the time of v0.1 prep found the field already contains substantively overlapping work: notably `pollockjj/ComfyUI-MultiGPU` and `turboderp-org/exllamav3`, both of which implement VRAM/DRAM offloading and per-component placement for consumer GPUs with adopted user bases. The three patterns formalized here (serial-handoff, fractional co-residency, VRAM→RAM overflow) are real and useful, but the marginal contribution of releasing them as a separate library — versus pointing users at one of those existing runtimes — was not clear at audit time.

The decision was: keep the scaffold, keep the trademark hygiene, do not release.

## What this means

- **No `pip install nvinx` for v0.1.** The PyPI `0.0.1` is a name-squat placeholder; it intentionally has no working code. Do not depend on it.
- **GitHub remote not pushed.** Local repo is the canonical state; remote `aki1770-del/nvinx` does not exist as of this status note.
- **No active development.** Issues / PRs / discussions are not currently being monitored.

## What would unpark it

- Material change in the field (the incumbents above deprecate or pivot away from this scope).
- A 4th pattern surfacing from real bench work that genuinely doesn't exist in the incumbents.
- A specific named user with a workload the incumbents can't serve and `nvinx`'s minimal-composable framing can.

## If you reached this repo and need this kind of scheduling

Try `pollockjj/ComfyUI-MultiGPU` or `turboderp-org/exllamav3` first. Both are actively maintained and have the patterns implemented at production scale.

## License posture

MIT remains. Trademark disclaimer in `README.md` (no NVIDIA / NGINX affiliation) remains. Fallback name `nvginx` is reserved on PyPI as an escape hatch if trademark concerns ever surface.
