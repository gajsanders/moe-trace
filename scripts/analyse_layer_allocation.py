from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from moe_trace.cache.simulator import _simulate_runtime_lru


TRACE_PATH = Path("results/workload_routing_trace.jsonl")
OUTPUT_PATH = Path("results/stage16_layer_allocation.json")

HELD_OUT = {
    "coding_05",
    "general_05",
    "math_05",
    "planning_05",
    "summary_05",
}

MIN_CAPACITY = 8
MAX_CAPACITY = 128

# The Stage 15 4 GiB streamlx baseline gave each of the
# 48 Qwen3 MoE layers 33 expert slots.
BASELINE_CAPACITY = 33


def load_events() -> list[dict]:
    with TRACE_PATH.open() as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def build_sequences(
    events: list[dict],
) -> dict[tuple[str, int], list[set[int]]]:

    grouped: dict[
        tuple[str, int],
        list[dict],
    ] = defaultdict(list)

    for event in events:
        if event["phase"] != "decode":
            continue

        grouped[
            (
                event["prompt_id"],
                int(event["layer"]),
            )
        ].append(event)

    sequences = {}

    for key, group in grouped.items():
        ordered = sorted(
            group,
            key=lambda event: event["token_index"],
        )

        sequences[key] = [
            set(event["expert_ids"])
            for event in ordered
        ]

    return sequences


def evaluate(
    *,
    sequences: dict[
        tuple[str, int],
        list[set[int]],
    ],
    prompt_ids: set[str],
    capacities: dict[int, int],
) -> tuple[int, int]:

    total_hits = 0
    total_misses = 0

    for (prompt_id, layer), sequence in sequences.items():

        if prompt_id not in prompt_ids:
            continue

        hits, misses = _simulate_runtime_lru(
            sequence,
            capacities[layer],
        )

        total_hits += hits
        total_misses += misses

    return total_hits, total_misses


def layer_miss_curve(
    *,
    sequences: dict[
        tuple[str, int],
        list[set[int]],
    ],
    prompt_ids: set[str],
    layer: int,
) -> dict[int, int]:

    relevant = [
        sequence
        for (prompt_id, sequence_layer), sequence
        in sequences.items()
        if (
            prompt_id in prompt_ids
            and sequence_layer == layer
        )
    ]

    curve = {}

    for capacity in range(
        MIN_CAPACITY,
        MAX_CAPACITY + 1,
    ):
        misses = 0

        for sequence in relevant:
            _, sequence_misses = _simulate_runtime_lru(
                sequence,
                capacity,
            )
            misses += sequence_misses

        curve[capacity] = misses

    return curve


def optimise_allocation(
    *,
    layers: list[int],
    curves: dict[int, dict[int, int]],
    total_capacity: int,
) -> dict[int, int]:
    """
    Find the exact minimum-training-miss allocation.

    Dynamic programming chooses one integer capacity for each
    layer while keeping the same total number of expert slots.

    Only training routing traces are used.
    """

    n_layers = len(layers)

    inf = np.iinfo(np.int64).max // 4

    dp = np.full(
        (n_layers + 1, total_capacity + 1),
        inf,
        dtype=np.int64,
    )

    choice = np.full(
        (n_layers + 1, total_capacity + 1),
        -1,
        dtype=np.int16,
    )

    dp[0, 0] = 0

    for i, layer in enumerate(layers, start=1):

        layers_left = n_layers - i

        for used_before in range(
            total_capacity + 1
        ):
            previous_cost = dp[
                i - 1,
                used_before,
            ]

            if previous_cost >= inf:
                continue

            max_here = min(
                MAX_CAPACITY,
                total_capacity
                - used_before
                - layers_left * MIN_CAPACITY,
            )

            if max_here < MIN_CAPACITY:
                continue

            for capacity in range(
                MIN_CAPACITY,
                max_here + 1,
            ):
                used = used_before + capacity

                cost = (
                    previous_cost
                    + curves[layer][capacity]
                )

                if cost < dp[i, used]:
                    dp[i, used] = cost
                    choice[i, used] = capacity

    if dp[n_layers, total_capacity] >= inf:
        raise RuntimeError(
            "No feasible layer allocation found."
        )

    allocation = {}
    remaining = total_capacity

    for i in range(
        n_layers,
        0,
        -1,
    ):
        capacity = int(
            choice[i, remaining]
        )

        if capacity < 0:
            raise RuntimeError(
                "Allocation reconstruction failed."
            )

        layer = layers[i - 1]
        allocation[layer] = capacity
        remaining -= capacity

    if remaining != 0:
        raise RuntimeError(
            "Allocation does not preserve total capacity."
        )

    return dict(sorted(allocation.items()))


def metrics(
    hits: int,
    misses: int,
) -> dict:

    requests = hits + misses

    return {
        "hits": hits,
        "misses": misses,
        "requests": requests,
        "hit_rate": (
            hits / requests
            if requests
            else 0.0
        ),
        "miss_rate": (
            misses / requests
            if requests
            else 0.0
        ),
    }


def print_comparison(
    name: str,
    equal: dict,
    optimised: dict,
) -> float:

    reduction = (
        (
            equal["misses"]
            - optimised["misses"]
        )
        / equal["misses"]
        if equal["misses"]
        else 0.0
    )

    print(f"\n{name}")
    print("-" * 72)

    print(
        f"{'Metric':<22}"
        f"{'Equal':>14}"
        f"{'Layer-aware':>16}"
        f"{'Delta':>16}"
    )

    print(
        f"{'Hit rate':<22}"
        f"{equal['hit_rate']:>13.2%} "
        f"{optimised['hit_rate']:>15.2%} "
        f"{optimised['hit_rate'] - equal['hit_rate']:>+14.2%}"
    )

    print(
        f"{'Miss rate':<22}"
        f"{equal['miss_rate']:>13.2%} "
        f"{optimised['miss_rate']:>15.2%} "
        f"{optimised['miss_rate'] - equal['miss_rate']:>+14.2%}"
    )

    print(
        f"{'Misses':<22}"
        f"{equal['misses']:>14,}"
        f"{optimised['misses']:>16,}"
        f"{optimised['misses'] - equal['misses']:>+16,}"
    )

    print(
        f"\nRelative miss reduction: "
        f"{reduction:.2%}"
    )

    return reduction


def main() -> None:

    events = load_events()
    sequences = build_sequences(events)

    prompt_ids = {
        prompt_id
        for prompt_id, _ in sequences
    }

    missing_holdout = HELD_OUT - prompt_ids

    if missing_holdout:
        raise RuntimeError(
            "Held-out prompts missing from trace: "
            + ", ".join(
                sorted(missing_holdout)
            )
        )

    train_ids = prompt_ids - HELD_OUT

    layers = sorted({
        layer
        for _, layer in sequences
    })

    if len(layers) != 48:
        raise RuntimeError(
            f"Expected 48 MoE layers; "
            f"found {len(layers)}."
        )

    total_capacity = (
        BASELINE_CAPACITY * len(layers)
    )

    print(
        "\nMoE Trace — Stage 16A "
        "Layer-Aware Cache Allocation"
    )
    print("=" * 72)

    print(
        f"Training prompts:  {len(train_ids)}"
    )
    print(
        f"Held-out prompts:  {len(HELD_OUT)}"
    )
    print(
        f"Layers:            {len(layers)}"
    )
    print(
        f"Equal capacity:    "
        f"{BASELINE_CAPACITY} experts/layer"
    )
    print(
        f"Total slots:       {total_capacity}"
    )
    print(
        f"Minimum/layer:     {MIN_CAPACITY}"
    )

    print(
        "\nBuilding training miss curves..."
    )

    curves = {
        layer: layer_miss_curve(
            sequences=sequences,
            prompt_ids=train_ids,
            layer=layer,
        )
        for layer in layers
    }

    print(
        "Optimising fixed total capacity..."
    )

    optimised_capacities = optimise_allocation(
        layers=layers,
        curves=curves,
        total_capacity=total_capacity,
    )

    equal_capacities = {
        layer: BASELINE_CAPACITY
        for layer in layers
    }

    print("\nOptimised layer capacities")
    print("-" * 72)

    for layer in layers:
        delta = (
            optimised_capacities[layer]
            - BASELINE_CAPACITY
        )

        print(
            f"Layer {layer:2d}: "
            f"{optimised_capacities[layer]:3d} "
            f"({delta:+d})"
        )

    values = list(
        optimised_capacities.values()
    )

    print(
        f"\nAllocation range: "
        f"{min(values)}–{max(values)}"
    )

    print(
        f"Allocation sum:   "
        f"{sum(values)}"
    )

    train_equal = metrics(
        *evaluate(
            sequences=sequences,
            prompt_ids=train_ids,
            capacities=equal_capacities,
        )
    )

    train_optimised = metrics(
        *evaluate(
            sequences=sequences,
            prompt_ids=train_ids,
            capacities=optimised_capacities,
        )
    )

    test_equal = metrics(
        *evaluate(
            sequences=sequences,
            prompt_ids=HELD_OUT,
            capacities=equal_capacities,
        )
    )

    test_optimised = metrics(
        *evaluate(
            sequences=sequences,
            prompt_ids=HELD_OUT,
            capacities=optimised_capacities,
        )
    )

    train_reduction = print_comparison(
        "TRAINING SET",
        train_equal,
        train_optimised,
    )

    test_reduction = print_comparison(
        "HELD-OUT SET",
        test_equal,
        test_optimised,
    )

    if test_reduction >= 0.10:
        decision = (
            "STRONG PASS — proceed to "
            "Stage 16B runtime validation"
        )
    elif test_reduction >= 0.05:
        decision = (
            "PASS — proceed to "
            "Stage 16B runtime validation"
        )
    else:
        decision = (
            "STOP — static layer allocation "
            "does not provide enough held-out headroom"
        )

    print("\nDECISION")
    print("=" * 72)
    print(decision)

    output = {
        "stage": "16A",
        "trace": str(TRACE_PATH),
        "policy": "runtime_lru",
        "selection": (
            "exact dynamic-programming allocation "
            "using training prompts only"
        ),
        "train_prompt_ids": sorted(train_ids),
        "held_out_prompt_ids": sorted(HELD_OUT),
        "baseline_capacity_per_layer": BASELINE_CAPACITY,
        "total_capacity": total_capacity,
        "minimum_capacity_per_layer": MIN_CAPACITY,
        "equal_capacities": {
            str(layer): capacity
            for layer, capacity
            in equal_capacities.items()
        },
        "optimised_capacities": {
            str(layer): capacity
            for layer, capacity
            in optimised_capacities.items()
        },
        "training": {
            "equal": train_equal,
            "layer_aware": train_optimised,
            "relative_miss_reduction": train_reduction,
        },
        "held_out": {
            "equal": test_equal,
            "layer_aware": test_optimised,
            "relative_miss_reduction": test_reduction,
        },
        "decision": decision,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            output,
            indent=2,
        )
        + "\n"
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
