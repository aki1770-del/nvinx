"""Tests for `nvinx.substrate` — v0.3.0a2 substrate-class gate.

Per Track C 2026-05-15 first-party measurement + Path B1 closure 2026-05-16:
- A1000 mobile substrate (16 SMs, 4 GB, sm_86) → "mobile" class
- A100 SXM4 40GB datacenter substrate (108 SMs, 40 GB, sm_80) → "datacenter" class
- v0.3.0a2 substrate-class gate emits RuntimeWarning when non-zero
  gamma_kernel_size is passed AND substrate is detected as datacenter-class

These tests verify the classifier logic + warning behavior using mocked
substrate info (does not require the test machine to be on a specific
substrate; uses the internal `_classify` function for unit testing).
"""

from __future__ import annotations

import warnings

from nvinx.substrate import (
    SubstrateInfo,
    _classify,
    detect_substrate,
    warn_if_datacenter_with_nonzero_gamma,
)

# ─── _classify unit tests ────────────────────────────────────────────────────


def test_classify_a100_sxm4_40gb_is_datacenter():
    """A100 SXM4 40GB: 108 SMs, 40 GB VRAM → datacenter."""
    assert _classify(sm_count=108, total_memory_mb=40 * 1024) == "datacenter"


def test_classify_a100_pcie_80gb_is_datacenter():
    """A100 PCIe 80GB: 108 SMs, 80 GB VRAM → datacenter."""
    assert _classify(sm_count=108, total_memory_mb=80 * 1024) == "datacenter"


def test_classify_h100_80gb_is_datacenter():
    """H100 80GB: 132 SMs, 80 GB VRAM → datacenter."""
    assert _classify(sm_count=132, total_memory_mb=80 * 1024) == "datacenter"


def test_classify_a6000_ampere_is_datacenter():
    """A6000 Ampere: 84 SMs, 48 GB VRAM → datacenter."""
    assert _classify(sm_count=84, total_memory_mb=48 * 1024) == "datacenter"


def test_classify_a10_24gb_is_datacenter():
    """A10: 72 SMs, 24 GB VRAM → datacenter."""
    assert _classify(sm_count=72, total_memory_mb=24 * 1024) == "datacenter"


def test_classify_v100_16gb_is_datacenter():
    """V100 16GB: 80 SMs, 16 GB VRAM → datacenter."""
    assert _classify(sm_count=80, total_memory_mb=16 * 1024) == "datacenter"


def test_classify_a1000_mobile_4gb_is_mobile():
    """A1000 Laptop: 16 SMs, 4 GB VRAM → mobile (the native v0.5 calibration substrate)."""
    assert _classify(sm_count=16, total_memory_mb=4 * 1024) == "mobile"


def test_classify_rtx_3070_mobile_is_mobile():
    """RTX 3070 mobile: 40 SMs, 8 GB VRAM → mobile."""
    assert _classify(sm_count=40, total_memory_mb=8 * 1024) == "mobile"


def test_classify_rtx_3080_mobile_is_mobile():
    """RTX 3080 mobile: 48 SMs, 16 GB VRAM → mobile (SM count at threshold)."""
    assert _classify(sm_count=48, total_memory_mb=16 * 1024) == "mobile"


def test_classify_rtx_4060_mobile_is_mobile():
    """RTX 4060 mobile: 24 SMs, 8 GB VRAM → mobile (small memory)."""
    assert _classify(sm_count=24, total_memory_mb=8 * 1024) == "mobile"


def test_classify_intermediate_returns_unknown():
    """Edge case: 56 SMs + 12 GB VRAM falls in neither bucket → unknown."""
    # 56 SMs > 48 (mobile threshold) but ≤ 64 (datacenter threshold)
    # 12 GB ≥ 12 GB (mobile threshold) but < 16 GB (datacenter threshold)
    assert _classify(sm_count=56, total_memory_mb=12 * 1024) == "unknown"


def test_classify_no_data_returns_unknown():
    assert _classify(sm_count=None, total_memory_mb=None) == "unknown"


def test_classify_only_high_sm_no_memory_returns_unknown():
    """108 SMs but no memory data → unknown (need both for datacenter classification)."""
    assert _classify(sm_count=108, total_memory_mb=None) == "unknown"


def test_classify_only_mobile_memory_returns_mobile():
    """4 GB VRAM with no SM info → mobile (small memory alone is sufficient)."""
    assert _classify(sm_count=None, total_memory_mb=4 * 1024) == "mobile"


# ─── detect_substrate integration test ───────────────────────────────────────


def test_detect_substrate_returns_substrate_info():
    """detect_substrate must always return a SubstrateInfo (may be 'unknown' if NVML absent)."""
    info = detect_substrate()
    assert isinstance(info, SubstrateInfo)
    assert info.substrate_class in ("datacenter", "mobile", "unknown")
    # name field is always populated
    assert info.name


# ─── warn_if_datacenter_with_nonzero_gamma behavior ──────────────────────────


def test_warn_skips_when_gamma_is_none():
    """No warning when gamma_kernel_size is None (V1 baseline; nothing to warn about)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_datacenter_with_nonzero_gamma(None)
    assert len(caught) == 0


def test_warn_skips_when_gamma_is_zero():
    """No warning when gamma_kernel_size is 0.0 (V1 baseline; nothing to warn about)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_datacenter_with_nonzero_gamma(0.0)
    assert len(caught) == 0


def test_warn_fires_on_datacenter_with_nonzero_gamma(monkeypatch):
    """Substrate-class gate: warning fires when γ ≠ 0 AND substrate is datacenter."""
    fake_datacenter = SubstrateInfo(
        name="NVIDIA A100-SXM4-40GB",
        sm_count=108,
        total_memory_mb=40960,
        compute_capability_major=8,
        compute_capability_minor=0,
        substrate_class="datacenter",
    )

    def fake_detect(device_index: int = 0) -> SubstrateInfo:
        return fake_datacenter

    monkeypatch.setattr("nvinx.substrate.detect_substrate", fake_detect)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_datacenter_with_nonzero_gamma(0.7456)
    assert len(caught) == 1
    msg = str(caught[0].message)
    assert "substrate-class gate" in msg
    assert "A100" in msg
    assert "0.7456" in msg


def test_warn_does_not_fire_on_mobile_with_nonzero_gamma(monkeypatch):
    """Mobile substrate: no warning even with non-zero γ (V5 helps on mobile)."""
    fake_mobile = SubstrateInfo(
        name="NVIDIA RTX A1000 Laptop GPU",
        sm_count=16,
        total_memory_mb=4096,
        compute_capability_major=8,
        compute_capability_minor=6,
        substrate_class="mobile",
    )

    def fake_detect(device_index: int = 0) -> SubstrateInfo:
        return fake_mobile

    monkeypatch.setattr("nvinx.substrate.detect_substrate", fake_detect)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_datacenter_with_nonzero_gamma(0.7456)
    assert len(caught) == 0


def test_warn_does_not_fire_on_unknown_substrate(monkeypatch):
    """Unknown substrate: no warning (operator can't be steered without classification)."""
    fake_unknown = SubstrateInfo(
        name="(pynvml not installed)",
        sm_count=None,
        total_memory_mb=None,
        compute_capability_major=None,
        compute_capability_minor=None,
        substrate_class="unknown",
    )

    def fake_detect(device_index: int = 0) -> SubstrateInfo:
        return fake_unknown

    monkeypatch.setattr("nvinx.substrate.detect_substrate", fake_detect)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warn_if_datacenter_with_nonzero_gamma(0.7456)
    assert len(caught) == 0


# ─── fit_gamma_kernel_size γ-fit precision warning (LB-B1-2 finding) ────────


def test_fit_gamma_warns_when_pairs_below_6():
    """fit_gamma_kernel_size emits RuntimeWarning when called with <6 pairs."""
    from nvinx.interference import (
        HardwareCoefficients,
        InterferenceProfile,
        fit_gamma_kernel_size,
    )

    profiles = {
        "model_a": InterferenceProfile(
            name="model_a",
            kernels=100,
            baseidle_ms=0.05,
            act_solo_ms=10.0,
            l2_saturation_pct=10.0,
            k_l2=0.0,
            theta=1.0,
        ),
        "model_b": InterferenceProfile(
            name="model_b",
            kernels=200,
            baseidle_ms=0.05,
            act_solo_ms=20.0,
            l2_saturation_pct=15.0,
            k_l2=0.0,
            theta=1.0,
        ),
    }
    hw = HardwareCoefficients(
        idlef_polynomial=(1.0, 0.0),
        powerp_linear=(1.0,),
        nominal_freq_mhz=1500.0,
        tdp_watts=100.0,
    )
    pair_measurements = [("model_a", "model_b", 13.0, 23.0)]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_gamma_kernel_size(profiles, pair_measurements, hw)

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings) >= 1
    msg = str(runtime_warnings[0].message)
    assert "Path B1 diagnostic" in msg
    assert "≥6 pairs" in msg or ">= 6 pairs" in msg or "<6" in msg or "< 6" in msg
