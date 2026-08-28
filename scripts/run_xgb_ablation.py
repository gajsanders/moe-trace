import json
from pathlib import Path

import numpy as np

from moe_trace.cache.simulator import simulate_policy
from moe_trace.cache.xgb_predictor import (
    FEATURE_NAMES,
    build_training_data,
    simulate_xgb_cache,
    train_xgb_predictor,
)


TRACE_PATH = Path(
    "results/workload_routing_trace.jsonl"
)


TEST_PROMPTS = {
    "coding_05",
    "general_05",
    "math_05",
    "planning_05",
    "summary_05",
}


# Focus first on the cache sizes where useful
# headroom still exists.
CAPACITIES = [
    8,
    16,
]


ABLATIONS = {
    "full": [],

    "no_expert_id": [
        "expert_id",
    ],

    "no_layer": [
        "layer",
    ],

    "no_router_scores": [
        "current_router_score",
        "previous_router_score",
    ],

    "no_markov_score": [
        "markov_score",
    ],

    "no_recency_history": [
        "previous_membership",
        "recent_4",
        "recent_8",
        "age_since_last_use",
        "historical_frequency",
    ],

    "no_workload": [
        "workload_coding",
        "workload_general",
        "workload_math",
        "workload_planning",
        "workload_summarisation",
    ],
}


def retained_indices(removed_names):
    removed = set(removed_names)

    return [
        index
        for index, name in enumerate(FEATURE_NAMES)
        if name not in removed
    ]


def main():

    with TRACE_PATH.open() as f:
        events = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    all_prompts = {
        event["prompt_id"]
        for event in events
    }

    train_prompts = (
        all_prompts - TEST_PROMPTS
    )

    print(
        "\nMoE Trace — XGBoost Feature Ablation"
    )
    print("=" * 78)

    print(
        f"Training prompts: {len(train_prompts)}"
    )

    print(
        f"Test prompts:     {len(TEST_PROMPTS)}"
    )

    print(
        "\nBuilding base training matrix..."
    )

    X_full, y_train = build_training_data(
        events,
        train_prompt_ids=train_prompts,
    )

    print(
        f"Training rows: {len(y_train):,}"
    )

    print(
        f"Full features: {X_full.shape[1]}"
    )

    print("\nFeature order:")

    for index, name in enumerate(FEATURE_NAMES):
        print(
            f"  {index:>2}: {name}"
        )

    # Held-out data for LRU / Markov / Oracle.
    test_events = [
        event
        for event in events
        if event["prompt_id"] in TEST_PROMPTS
    ]

    baseline_results = {}

    for capacity in CAPACITIES:

        baseline_results[capacity] = {
            "lru": simulate_policy(
                test_events,
                policy="lru",
                capacity=capacity,
            ),
            "markov": simulate_policy(
                test_events,
                policy="markov",
                capacity=capacity,
            ),
            "oracle": simulate_policy(
                test_events,
                policy="oracle",
                capacity=capacity,
            ),
        }

    experiment_results = {}

    for experiment_name, removed_features in ABLATIONS.items():

        indices = retained_indices(
            removed_features
        )

        X_train = X_full[:, indices]

        print("\n" + "=" * 78)

        print(
            f"EXPERIMENT: {experiment_name}"
        )

        print("-" * 78)

        if removed_features:
            print(
                "Removed: "
                + ", ".join(removed_features)
            )
        else:
            print("Removed: none")

        print(
            f"Features retained: "
            f"{len(indices)} / {len(FEATURE_NAMES)}"
        )

        print("Training...")

        model = train_xgb_predictor(
            X_train,
            y_train,
        )

        print("Training complete.")

        experiment_results[
            experiment_name
        ] = {}

        for capacity in CAPACITIES:

            hits, misses = simulate_xgb_cache(
                events,
                test_prompt_ids=TEST_PROMPTS,
                model=model,
                capacity=capacity,
                feature_indices=indices,
            )

            requests = hits + misses
            hit_rate = hits / requests

            experiment_results[
                experiment_name
            ][capacity] = hit_rate

            print(
                f"capacity={capacity:<2} "
                f"hit_rate={hit_rate:>6.2%} "
                f"hits={hits:>7} "
                f"misses={misses:>7}"
            )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\n" + "=" * 78)
    print("ABLATION SUMMARY")
    print("=" * 78)

    full_results = experiment_results["full"]

    for capacity in CAPACITIES:

        baseline = baseline_results[capacity]

        lru_rate = baseline["lru"].hit_rate
        markov_rate = baseline["markov"].hit_rate
        oracle_rate = baseline["oracle"].hit_rate

        full_rate = full_results[capacity]

        print(
            f"\nCACHE CAPACITY: {capacity}"
        )

        print("-" * 78)

        print(
            f"LRU:       {lru_rate:>6.2%}"
        )

        print(
            f"Markov:    {markov_rate:>6.2%}"
        )

        print(
            f"Full XGB:  {full_rate:>6.2%}"
        )

        print(
            f"Oracle:    {oracle_rate:>6.2%}"
        )

        print()

        ranked = []

        for experiment_name, results in experiment_results.items():

            if experiment_name == "full":
                continue

            ablated_rate = results[capacity]

            drop = (
                full_rate - ablated_rate
            )

            ranked.append(
                (
                    experiment_name,
                    ablated_rate,
                    drop,
                )
            )

        ranked.sort(
            key=lambda item: item[2],
            reverse=True,
        )

        print(
            "Ablation impact "
            "(positive drop = feature group helps):"
        )

        for name, rate, drop in ranked:

            print(
                f"{name:<22} "
                f"hit_rate={rate:>6.2%}  "
                f"drop={drop:+.2%}"
            )


if __name__ == "__main__":
    main()
