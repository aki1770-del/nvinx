"""WorkloadScheduler — high-level orchestration layer (planned for v0.2).

v0.1 exposes the three scheduling patterns directly from ``nvinx.patterns``.
A scheduler that sequences multiple patterns across a run — e.g. serial_handoff
during an overnight window, then fractional_coresidency for the ranking phase —
is slated for v0.2 once the pattern API stabilizes against real workloads.
"""
