from __future__ import annotations

import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from mlx_lm import load


MODEL = "mlx-community/Qwen3-30B-A3B-4bit"
TRACE_PATH = Path("results/workload_routing_trace.jsonl")

LAYERS = [0, 7, 18, 31, 47]

ROUNDS = 8
SEED = 42


def load_sequences():
    grouped = defaultdict(list)

    with TRACE_PATH.open() as handle:
        for line in handle:
            event = json.loads(line)

            if event["phase"] != "decode":
                continue

            if event["layer"] not in LAYERS:
                continue

            grouped[
                (
                    event["prompt_id"],
                    event["layer"],
                )
            ].append(event)

    sequences = {}

    for key, events in grouped.items():
        sequences[key] = sorted(
            events,
            key=lambda event: event["token_index"],
        )

    return sequences


def make_indices(experts):
    return mx.array(
        experts,
        dtype=mx.int32,
    ).reshape(1, 1, 8)


def oracle_next_last_order(
    current_experts,
    next_experts,
):
    next_set = set(next_experts)

    cold = [
        expert
        for expert in current_experts
        if expert not in next_set
    ]

    reused = [
        expert
        for expert in current_experts
        if expert in next_set
    ]

    return cold + reused


def permutation_to_restore(
    original,
    reordered,
):
    """
    Return positions that restore reordered output
    to the original expert order.
    """

    position = {
        expert: index
        for index, expert in enumerate(reordered)
    }

    return [
        position[expert]
        for expert in original
    ]


def execute_reordered(
    switch_mlp,
    x,
    original_experts,
    execution_experts,
):
    indices = make_indices(
        execution_experts
    )

    y = switch_mlp(
        x,
        indices,
    )

    if execution_experts != original_experts:
        restore = permutation_to_restore(
            original_experts,
            execution_experts,
        )

        restore_indices = mx.array(
            restore,
            dtype=mx.int32,
        )

        y = mx.take(
            y,
            restore_indices,
            axis=-2,
        )

    return y


def warm_layer(
    switch_mlp,
    x,
):
    for start in range(0, 128, 8):
        experts = list(
            range(start, start + 8)
        )

        y = switch_mlp(
            x,
            make_indices(experts),
        )

        mx.eval(y)


def verify_equivalence(
    switch_mlp,
    x,
    original,
    reordered,
):
    y_original = switch_mlp(
        x,
        make_indices(original),
    )

    y_reordered = execute_reordered(
        switch_mlp,
        x,
        original,
        reordered,
    )

    mx.eval(
        y_original,
        y_reordered,
    )

    difference = mx.max(
        mx.abs(
            y_original.astype(mx.float32)
            - y_reordered.astype(mx.float32)
        )
    )

    mx.eval(difference)

    return float(difference.item())


def benchmark_sequence(
    switch_mlp,
    x,
    events,
    condition,
):
    latencies = []

    for index, event in enumerate(events):
        original = list(
            event["expert_ids"]
        )

        if condition == "natural":
            execution = original

        elif condition == "sorted":
            execution = sorted(
                original
            )

        elif condition == "oracle_next_last":
            if index + 1 < len(events):
                following = events[
                    index + 1
                ]["expert_ids"]

                execution = (
                    oracle_next_last_order(
                        original,
                        following,
                    )
                )
            else:
                execution = original

        else:
            raise ValueError(
                f"Unknown condition: {condition}"
            )

        start = time.perf_counter()

        y = execute_reordered(
            switch_mlp,
            x,
            original,
            execution,
        )

        mx.eval(y)

        elapsed_ms = (
            time.perf_counter() - start
        ) * 1000.0

        latencies.append(
            elapsed_ms
        )

    return latencies


def main():
    print()
    print(
        "MoE Trace — Expert Order Steering Benchmark"
    )
    print("=" * 78)

    print(f"Model:  {MODEL}")
    print(f"Trace:  {TRACE_PATH}")
    print(
        "Layers: "
        + ", ".join(map(str, LAYERS))
    )
    print(f"Rounds: {ROUNDS}")

    sequences = load_sequences()

    print(
        "Prompt-layer sequences: "
        f"{len(sequences)}"
    )

    print()
    print("Loading model...")

    model, _ = load(MODEL)

    print("Model loaded.")

    conditions = [
        "natural",
        "sorted",
        "oracle_next_last",
    ]

    all_results = {
        layer: {
            condition: []
            for condition in conditions
        }
        for layer in LAYERS
    }

    rng = random.Random(SEED)

    equivalence_checked = False

    for layer in LAYERS:
        print()
        print("=" * 78)
        print(f"LAYER {layer}")
        print("=" * 78)

        switch_mlp = (
            model.model.layers[
                layer
            ].mlp.switch_mlp
        )

        hidden_size = (
            switch_mlp.gate_proj.input_dims
        )

        x = mx.random.normal(
            shape=(1, 1, hidden_size)
        ).astype(mx.bfloat16)

        mx.eval(x)

        warm_layer(
            switch_mlp,
            x,
        )

        layer_sequences = [
            (prompt_id, events)
            for (
                prompt_id,
                sequence_layer,
            ), events in sequences.items()
            if sequence_layer == layer
        ]

        layer_sequences.sort()

        if (
            not equivalence_checked
            and layer_sequences
        ):
            _, example_events = (
                layer_sequences[0]
            )

            if len(example_events) >= 2:
                original = list(
                    example_events[0][
                        "expert_ids"
                    ]
                )

                reordered = (
                    oracle_next_last_order(
                        original,
                        example_events[1][
                            "expert_ids"
                        ],
                    )
                )

                max_diff = verify_equivalence(
                    switch_mlp,
                    x,
                    original,
                    reordered,
                )

                print(
                    "Semantic equivalence "
                    f"max abs diff: {max_diff}"
                )

                equivalence_checked = True

        for prompt_id, events in layer_sequences:

            prompt_results = {
                condition: []
                for condition in conditions
            }

            for round_index in range(
                ROUNDS
            ):
                order = (
                    conditions.copy()
                )

                rng.shuffle(order)

                for condition in order:
                    values = (
                        benchmark_sequence(
                            switch_mlp,
                            x,
                            events,
                            condition,
                        )
                    )

                    prompt_results[
                        condition
                    ].extend(values)

            medians = {
                condition:
                statistics.median(
                    prompt_results[
                        condition
                    ]
                )
                for condition in conditions
            }

            for condition in conditions:
                all_results[
                    layer
                ][condition].extend(
                    prompt_results[
                        condition
                    ]
                )

            natural = medians[
                "natural"
            ]

            sorted_delta = (
                (
                    medians["sorted"]
                    - natural
                )
                / natural
                * 100.0
            )

            oracle_delta = (
                (
                    medians[
                        "oracle_next_last"
                    ]
                    - natural
                )
                / natural
                * 100.0
            )

            print(
                f"{prompt_id:<18} "
                f"natural={natural:.3f}  "
                f"sorted={sorted_delta:+.2f}%  "
                f"oracle={oracle_delta:+.2f}%"
            )

    print()
    print("=" * 78)
    print("LAYER SUMMARY")
    print("=" * 78)

    oracle_deltas = []

    for layer in LAYERS:
        natural = statistics.median(
            all_results[
                layer
            ]["natural"]
        )

        sorted_value = statistics.median(
            all_results[
                layer
            ]["sorted"]
        )

        oracle = statistics.median(
            all_results[
                layer
            ]["oracle_next_last"]
        )

        sorted_delta = (
            (sorted_value - natural)
            / natural
            * 100.0
        )

        oracle_delta = (
            (oracle - natural)
            / natural
            * 100.0
        )

        oracle_deltas.append(
            oracle_delta
        )

        print(
            f"Layer {layer:>2}: "
            f"natural={natural:.3f} ms  "
            f"sorted={sorted_delta:+.2f}%  "
            f"oracle={oracle_delta:+.2f}%"
        )

    print()
    print("=" * 78)
    print("AGGREGATE")
    print("=" * 78)

    print(
        "Median oracle-next-last delta: "
        f"{statistics.median(oracle_deltas):+.2f}%"
    )

    faster_layers = sum(
        delta < 0
        for delta in oracle_deltas
    )

    print(
        "Layers where oracle ordering "
        "was faster: "
        f"{faster_layers}/{len(LAYERS)}"
    )

    print()
    print("=" * 78)
    print("DECISION RULE")
    print("=" * 78)

    print(
        "A negative delta means the "
        "order-steering policy was faster."
    )

    print(
        "If oracle ordering does not produce "
        "a stable improvement, stop this branch."
    )

    print(
        "If oracle ordering produces a stable "
        "improvement, test a deployable predictor next."
    )


if __name__ == "__main__":
    main()
