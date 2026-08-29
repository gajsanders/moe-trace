from __future__ import annotations

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock


MODEL = "mlx-community/Qwen3-30B-A3B-4bit"


def array_bytes(array) -> int:
    try:
        return int(array.nbytes)
    except (AttributeError, TypeError):
        return 0


def main():
    print("\nMoE Trace — MLX Runtime Inspection")
    print("=" * 72)

    print(f"Loading model: {MODEL}")
    model, tokenizer = load(MODEL)

    print("Model loaded.")
    print(f"MLX device: {mx.default_device()}")

    layers = model.model.layers

    moe_blocks = []

    for layer_index, layer in enumerate(layers):
        mlp = layer.mlp

        print(
            f"Layer {layer_index:>2}: "
            f"{type(mlp).__name__}"
        )

        if isinstance(mlp, Qwen3MoeSparseMoeBlock):
            moe_blocks.append(
                (layer_index, mlp)
            )

    print()
    print("=" * 72)
    print("MOE SUMMARY")
    print("=" * 72)

    print(f"Decoder layers: {len(layers)}")
    print(f"MoE blocks found: {len(moe_blocks)}")

    if not moe_blocks:
        print("No MoE blocks found.")
        return

    first_layer_index, first_block = moe_blocks[0]

    print(
        f"Experts per MoE layer: "
        f"{first_block.num_experts}"
    )

    print(
        f"Experts selected per token: "
        f"{first_block.top_k}"
    )

    print(
        f"First MoE layer: "
        f"{first_layer_index}"
    )

    print()
    print("SwitchGLU parameters")
    print("-" * 72)

    layer_sizes = []

    for layer_index, block in moe_blocks:
        parameters = tree_flatten(
            block.switch_mlp.parameters()
        )

        total_bytes = 0

        print(f"\nLayer {layer_index}")

        for name, parameter in parameters:
            if not isinstance(parameter, mx.array):
                continue

            size = array_bytes(parameter)
            total_bytes += size

            print(
                f"  {name:<32} "
                f"shape={str(parameter.shape):<24} "
                f"dtype={str(parameter.dtype):<10} "
                f"bytes={size:,}"
            )

        layer_sizes.append(total_bytes)

        print(
            f"  SwitchGLU total: "
            f"{total_bytes / (1024 ** 2):.2f} MiB"
        )

    total_bytes = sum(layer_sizes)

    print()
    print("=" * 72)
    print("STORAGE SUMMARY")
    print("=" * 72)

    print(
        f"Total SwitchGLU storage: "
        f"{total_bytes / (1024 ** 3):.2f} GiB"
    )

    mean_layer_bytes = (
        total_bytes / len(layer_sizes)
    )

    print(
        f"Mean per MoE layer: "
        f"{mean_layer_bytes / (1024 ** 2):.2f} MiB"
    )

    estimated_expert_bytes = (
        mean_layer_bytes
        / first_block.num_experts
    )

    print(
        f"Approximate storage per expert: "
        f"{estimated_expert_bytes / (1024 ** 2):.2f} MiB"
    )

    print()
    print(
        "This measures stored expert parameters."
    )

    print(
        "It does not yet prove that MLX "
        "evicts or reloads experts."
    )


if __name__ == "__main__":
    main()