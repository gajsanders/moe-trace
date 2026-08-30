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
        events = sorted(
            events,
            key=lambda event: event["token_index"],
        )

        sequences[key] = [
            event["expert_ids"]
            for event in events
        ]

    return sequences


def to_indices(experts):
    return mx.array(
        experts,
        dtype=mx.int32,
    ).reshape(1, 1, 8)


def run_sequence(
    switch_mlp,
    x,
    sequence,
):
    latencies = []

    for experts in sequence:
        indices = to_indices(experts)

        start = time.perf_counter()

        output = switch_mlp(
            x,
            indices,
        )

        mx.eval(output)

        elapsed = (
            time.perf_counter() - start
        ) * 1000.0

        latencies.append(elapsed)

    return latencies


def warm_layer(
    switch_mlp,
    x,
):
    # Touch all experts before measurement.
    # This reduces first-use effects.
    for start in range(0, 128, 8):
        indices = to_indices(
            list(range(start, start + 8))
        )

        output = switch_mlp(
            x,
            indices,
        )

        mx.eval(output)


def main():
    print()
    print(
        "MoE Trace — Real Routing Locality Replay"
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
        f"Prompt-layer sequences: "
        f"{len(sequences)}"
    )

    if not sequences:
        raise RuntimeError(
            "No routing sequences found."
        )

    print()
    print("Loading model...")

    model, _ = load(MODEL)

    print("Model loaded.")

    rng = random.Random(SEED)

    results = {
        layer: {
            "natural": [],
            "shuffled": [],
        }
        for layer in LAYERS
    }

    paired_prompt_results = []

    for layer in LAYERS:
        print()
        print("=" * 78)
        print(f"LAYER {layer}")
        print("=" * 78)

        block = model.model.layers[layer].mlp
        switch_mlp = block.switch_mlp

        hidden_size = (
            switch_mlp.gate_proj.input_dims
        )

        x = mx.random.normal(
            shape=(1, 1, hidden_size)
        ).astype(mx.bfloat16)

        mx.eval(x)

        layer_sequences = [
            (prompt_id, sequence)
            for (prompt_id, sequence_layer), sequence
            in sequences.items()
            if sequence_layer == layer
        ]

        layer_sequences.sort()

        print(
            f"Prompts: {len(layer_sequences)}"
        )

        warm_layer(
            switch_mlp,
            x,
        )

        for prompt_id, natural_sequence in layer_sequences:

            condition_latencies = {
                "natural": [],
                "shuffled": [],
            }

            for round_index in range(ROUNDS):

                shuffled_sequence = (
                    natural_sequence.copy()
                )

                shuffle_rng = random.Random(
                    SEED
                    + layer * 10000
                    + round_index * 100
                    + sum(ord(c) for c in prompt_id)
                )

                shuffle_rng.shuffle(
                    shuffled_sequence
                )

                conditions = [
                    (
                        "natural",
                        natural_sequence,
                    ),
                    (
                        "shuffled",
                        shuffled_sequence,
                    ),
                ]

                rng.shuffle(conditions)

                for name, sequence in conditions:

                    latencies = run_sequence(
                        switch_mlp,
                        x,
                        sequence,
                    )

                    condition_latencies[
                        name
                    ].extend(latencies)

            natural_median = statistics.median(
                condition_latencies["natural"]
            )

            shuffled_median = statistics.median(
                condition_latencies["shuffled"]
            )

            delta_percent = (
                (
                    shuffled_median
                    - natural_median
                )
                / natural_median
                * 100.0
            )

            results[layer]["natural"].extend(
                condition_latencies["natural"]
            )

            results[layer]["shuffled"].extend(
                condition_latencies["shuffled"]
            )

            paired_prompt_results.append(
                (
                    layer,
                    prompt_id,
                    natural_median,
                    shuffled_median,
                    delta_percent,
                )
            )

            print(
                f"{prompt_id:<18} "
                f"natural={natural_median:.3f} ms  "
                f"shuffled={shuffled_median:.3f} ms  "
                f"delta={delta_percent:+.2f}%"
            )

    print()
    print("=" * 78)
    print("LAYER SUMMARY")
    print("=" * 78)

    layer_deltas = []

    for layer in LAYERS:
        natural = statistics.median(
            results[layer]["natural"]
        )

        shuffled = statistics.median(
            results[layer]["shuffled"]
        )

        delta = (
            (shuffled - natural)
            / natural
            * 100.0
        )

        layer_deltas.append(delta)

        print(
            f"Layer {layer:>2}: "
            f"natural={natural:.3f} ms  "
            f"shuffled={shuffled:.3f} ms  "
            f"delta={delta:+.2f}%"
        )

    print()
    print("=" * 78)
    print("PAIRED PROMPT SUMMARY")
    print("=" * 78)

    positive_prompts = sum(
        delta > 0
        for _, _, _, _, delta
        in paired_prompt_results
    )

    total_prompts = len(
        paired_prompt_results
    )

    prompt_deltas = [
        delta
        for _, _, _, _, delta
        in paired_prompt_results
    ]

    print(
        "Prompt-layer pairs where shuffled "
        "was slower: "
        f"{positive_prompts}/{total_prompts}"
    )

    print(
        "Median paired shuffled penalty: "
        f"{statistics.median(prompt_deltas):+.2f}%"
    )

    print(
        "Median layer-level shuffled penalty: "
        f"{statistics.median(layer_deltas):+.2f}%"
    )

    print()
    print("=" * 78)
    print("INTERPRETATION LIMIT")
    print("=" * 78)

    print(
        "Both conditions use exactly the same "
        "recorded expert selections."
    )

    print(
        "Only their temporal order changes."
    )

    print(
        "This benchmark replays real routing, "
        "but it is still an isolated SwitchGLU test."
    )

    print(
        "It does not yet measure end-to-end "
        "token generation speed."
    )


if __name__ == "__main__":
    main()

