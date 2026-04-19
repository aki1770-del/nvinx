"""Pattern C — ram_overflow: spill layers from VRAM to system RAM.

Scenario: a 6 GB model won't fit in a 4 GB card. With HuggingFace `accelerate`'s
`device_map="auto"`, layers that don't fit in VRAM execute from system RAM. The
tradeoff is a 3-5x slowdown, which is often better than not running the model.
"""

from nvinx.catalog import HardwareSpec, ModelSpec, Residency
from nvinx.patterns import ram_overflow


def main() -> None:
    hw = HardwareSpec(vram_gb=4.0, ram_gb=32.0, cpu_cores=8)

    big_model = ModelSpec(
        name="big_folder",
        vram_gb=6.0,
        residency=Residency.GPU_RAM_OVERFLOW,
        ram_overflow_supported=True,
        ram_gb_needed=10.0,
    )

    hint = ram_overflow(big_model, hw)

    print(f"device_map         : {hint['device_map']}")
    print(f"max_memory         : {hint['max_memory']}")
    print(f"estimated_slowdown : {hint['estimated_slowdown']}")
    print(f"notes              : {hint['notes']}")

    # Pass these into your runtime, e.g.:
    #   from transformers import AutoModelForXxx
    #   AutoModelForXxx.from_pretrained(
    #       "model-name",
    #       device_map=hint["device_map"],
    #       max_memory=hint["max_memory"],
    #   )


if __name__ == "__main__":
    main()
