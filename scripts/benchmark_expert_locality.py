from __future__ import annotations

import random
import statistics
import time

import mlx.core as mx
from mlx_lm import load


MODEL = "mlx-community/Qwen3-30B-A3B-4bit"

LAYER_INDICES = [0, 7, 18, 31, 47]

WARMUP_CALLS = 50
MEASURE_CALLS = 250
ROUNDS = 8

SEED = 42


def make_indices(experts: list[int]) -> mx.array:
    return mx.array(
        experts,
        dtype=mx.int32,
    ).reshape(1, 1, 8)


def build_patterns() -> dict[str, list[mx.array]]:
    same_group = list(range(0, 8))

    group_a = list(range(0, 8))
    group_b = list(range(8, 16))

    all_groups = [
        list(range(start, start + 8))
        for start in range(0, 128, 8)
    ]

    total_calls = (
        WARMUP_CALLS
        + MEASURE_CALLS
    )

    return {
        "same_8": [
            make_indices(same_group)
            for _ in range(total_calls)
        ],
        "alternate_16": [
            make_indices(
                group_a if i % 2 == 0 else group_b
            )
            for i in range(total_calls)
        ],
        "rotate_128": [
            make_indices(
                all_groups[i % len(all_groups)]
            )
            for i in range(total_calls)
        ],
    }


def run_condition(
    switch_mlp,
    x: mx.array,
    indices_sequence: list[mx.array],
) -> list[float]:

    for indices in indices_sequence[:WARMUP_CALLS]:
        output = switch_mlp(
            x,
            indices,
        )

        mx.eval(output)

    latencies_ms = []

    measured = indices_sequence[
        WARMUP_CALLS:
        WARMUP_CALLS + MEASURE_CALLS
    ]

    for indices in measured:
        start = time.perf_counter()

        output = switch_mlp(
            x,
            indices,
        )

        mx.eval(output)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        latencies_ms.append(
            elapsed_ms
        )

    return latencies_ms


def percentile(
    values: list[float],
    percentile_value: float,
) -> float:

    ordered = sorted(values)

    index = int(
        round(
            (len(ordered) - 1)
            * percentile_value
        )
    )

    return ordered[index]


def benchmark_layer(
    *,
    model,
    layer_index: int,
    patterns: dict[str, list[mx.array]],
) -> dict[str, dict[str, float]]:

    block = model.model.layers[
        layer_index
    ].mlp

    switch_mlp = block.switch_mlp

    hidden_size = (
        switch_mlp.gate_proj.input_dims
    )

    x = mx.random.normal(
        shape=(1, 1, hidden_size)
    ).astype(mx.bfloat16)

    mx.eval(x)

    results: dict[
        str,
        list[float],
    ] = {
        name: []
        for name in patterns
    }

    condition_names = list(
        patterns.keys()
    )

    rng = random.Random(
        SEED + layer_index
    )

    print()
    print("=" * 78)
    print(f"LAYER {layer_index}")
    print("=" * 78)

    for round_index in range(ROUNDS):

        order = condition_names.copy()
        rng.shuffle(order)

        print(
            f"Round {round_index + 1}/{ROUNDS}: "
            + ", ".join(order)
        )

        for condition in order:

            latencies = run_condition(
                switch_mlp,
                x,
                patterns[condition],
            )

            results[
                condition
            ].extend(latencies)

            print(
                f"  {condition:<14} "
                f"median="
                f"{statistics.median(latencies):>7.3f} ms"
            )

    summary = {}

    for condition in condition_names:

        values = results[
            condition
        ]

        summary[condition] = {
            "median": statistics.median(values),
            "mean": statistics.mean(values),
            "p95": percentile(
                values,
                0.95,
            ),
        }

    return summary


def main():
    print(
        "\nMoE Trace — Cross-Layer Expert Locality Benchmark"
    )

    print("=" * 78)

    print(f"Model:       {MODEL}")
    print(
        "Layers:      "
        + ", ".join(
            str(layer)
            for layer in LAYER_INDICES
        )
    )
    print(f"Warmup:      {WARMUP_CALLS}")
    print(f"Measured:    {MEASURE_CALLS}")
    print(f"Rounds:      {ROUNDS}")

    print("\nLoading model...")

    model, _ = load(MODEL)

    print("Model loaded.")

    patterns = build_patterns()

    all_summaries = {}

    for layer_index in LAYER_INDICES:

        all_summaries[
            layer_index
        ] = benchmark_layer(
            model=model,
            layer_index=layer_index,
            patterns=patterns,
        )

    print()
    print("=" * 78)
    print("CROSS-LAYER SUMMARY")
    print("=" * 78)

    print(
        f"{'Layer':>5}  "
        f"{'Same 8':>9}  "
        f"{'Alt 16':>9}  "
        f"{'Rotate 128':>11}  "
        f"{'Alt Δ%':>8}  "
        f"{'Rotate Δ%':>10}"
    )

    print("-" * 78)

    rotate_deltas = []
    alternate_deltas = []

    for layer_index in LAYER_INDICES:

        summary = all_summaries[
            layer_index
        ]

        same = summary[
            "same_8"
        ]["median"]

        alternate = summary[
            "alternate_16"
        ]["median"]

        rotate = summary[
            "rotate_128"
        ]["median"]

        alternate_delta = (
            (alternate - same)
            / same
            * 100.0
        )

        rotate_delta = (
            (rotate - same)
            / same
            * 100.0
        )

        alternate_deltas.append(
            alternate_delta
        )

        rotate_deltas.append(
            rotate_delta
        )

        print(
            f"{layer_index:>5}  "
            f"{same:>8.3f} ms  "
            f"{alternate:>8.3f} ms  "
            f"{rotate:>10.3f} ms  "
            f"{alternate_delta:>+7.2f}%  "
            f"{rotate_delta:>+9.2f}%"
        )

    print()
    print("=" * 78)
    print("AGGREGATE")
    print("=" * 78)

    print(
        "Median alternate_16 penalty: "
        f"{statistics.median(alternate_deltas):+.2f}%"
    )

    print(
        "Median rotate_128 penalty:   "
        f"{statistics.median(rotate_deltas):+.2f}%"
    )

    positive_rotate = sum(
        delta > 0
        for delta in rotate_deltas
    )

    positive_alternate = sum(
        delta > 0
        for delta in alternate_deltas
    )

    print(
        "Layers where alternate_16 "
        "was slower than same_8: "
        f"{positive_alternate}/{len(LAYER_INDICES)}"
    )

    print(
        "Layers where rotate_128 "
        "was slower than same_8: "
        f"{positive_rotate}/{len(LAYER_INDICES)}"
    )

    print()
    print("=" * 78)
    print("INTERPRETATION LIMIT")
    print("=" * 78)

    print(
        "This benchmark measures expert-access locality "
        "inside resident MLX SwitchGLU layers."
    )

    print(
        "It does not measure SSD loading, "
        "expert offloading, or an explicit software cache."
    )

    print(
        "A consistent positive timing penalty across layers "
        "would support a real resident-memory locality effect."
    )


if __name__ == "__main__":
    main()