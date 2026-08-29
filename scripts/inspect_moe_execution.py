from __future__ import annotations

from mlx_lm import load


MODEL = "mlx-community/Qwen3-30B-A3B-4bit"


def main():
    print("\nMoE Trace — MLX Expert Execution Inspection")
    print("=" * 72)

    model, _ = load(MODEL)

    block = model.model.layers[0].mlp
    switch = block.switch_mlp

    print(f"MoE block type:       {type(block).__name__}")
    print(f"SwitchGLU type:       {type(switch).__name__}")

    print()
    print("Projection implementations")
    print("-" * 72)

    for name in [
        "gate_proj",
        "up_proj",
        "down_proj",
    ]:
        projection = getattr(switch, name)

        print(
            f"{name:<12}: "
            f"{type(projection).__module__}."
            f"{type(projection).__name__}"
        )

        if hasattr(projection, "num_experts"):
            print(
                f"  num_experts: "
                f"{projection.num_experts}"
            )

        if hasattr(projection, "bits"):
            print(
                f"  quantization bits: "
                f"{projection.bits}"
            )

        if hasattr(projection, "group_size"):
            print(
                f"  group size: "
                f"{projection.group_size}"
            )

    print()
    print("=" * 72)
    print("INTERPRETATION")
    print("=" * 72)

    projection_types = {
        type(switch.gate_proj).__name__,
        type(switch.up_proj).__name__,
        type(switch.down_proj).__name__,
    }

    print(
        "Projection types: "
        + ", ".join(sorted(projection_types))
    )

    print()
    print(
        "If the projections are QuantizedSwitchLinear, "
        "expert computation uses the selected expert indices "
        "through MLX quantized gather matrix multiplication."
    )

    print(
        "This shows sparse expert computation."
    )

    print(
        "It does not by itself show expert eviction, "
        "SSD loading, or a software expert cache."
    )


if __name__ == "__main__":
    main()
