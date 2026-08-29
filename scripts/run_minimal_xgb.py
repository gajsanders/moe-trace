import json
from pathlib import Path

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

CAPACITIES = [
    8,
    16,
]


FEATURE_SETS = {
    "markov_only": [
        "markov_score",
    ],

    "markov_router": [
        "markov_score",
        "current_router_score",
        "previous_router_score",
    ],

    "markov_workload": [
        "markov_score",
        "workload_coding",
        "workload_general",
        "workload_math",
        "workload_planning",
        "workload_summarisation",
    ],

    "markov_router_workload": [
        "markov_score",
        "current_router_score",
        "previous_router_score",
        "workload_coding",
        "workload_general",
        "workload_math",
        "workload_planning",
        "workload_summarisation",
    ],

    "full": FEATURE_NAMES,
}


def feature_indices(
    feature_names: list[str],
) -> list[int]:

    return [
        FEATURE_NAMES.index(name)
        for name in feature_names
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
        "\nMoE Trace — Minimal XGBoost Experiment"
    )
    print("=" * 78)

    print(
        f"Training prompts: {len(train_prompts)}"
    )

    print(
        f"Test prompts:     {len(TEST_PROMPTS)}"
    )

    print(
        "\nBuilding training matrix..."
    )

    X_full, y_train = build_training_data(
        events,
        train_prompt_ids=train_prompts,
    )

    print(
        f"Training rows: {len(y_train):,}"
    )

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

    model_results = {}

    for model_name, features in FEATURE_SETS.items():

        indices = feature_indices(features)

        X_train = X_full[:, indices]

        print()
        print("=" * 78)
        print(f"MODEL: {model_name}")
        print("-" * 78)

        print(
            "Features: "
            + ", ".join(features)
        )

        print(
            f"Feature count: {len(indices)}"
        )

        print("Training...")

        model = train_xgb_predictor(
            X_train,
            y_train,
        )

        print("Training complete.")

        model_results[model_name] = {}

        for capacity in CAPACITIES:

            hits, misses = simulate_xgb_cache(
                events,
                test_prompt_ids=TEST_PROMPTS,
                model=model,
                capacity=capacity,
                feature_indices=indices,
            )

            total = hits + misses
            hit_rate = hits / total

            model_results[
                model_name
            ][capacity] = {
                "hit_rate": hit_rate,
                "hits": hits,
                "misses": misses,
            }

            print(
                f"capacity={capacity:<2} "
                f"hit_rate={hit_rate:>6.2%} "
                f"hits={hits:>7} "
                f"misses={misses:>7}"
            )

    print()
    print("=" * 78)
    print("MINIMAL MODEL SUMMARY")
    print("=" * 78)

    for capacity in CAPACITIES:

        baseline = baseline_results[capacity]

        lru_rate = baseline["lru"].hit_rate
        markov_rate = baseline["markov"].hit_rate
        oracle_rate = baseline["oracle"].hit_rate

        full_rate = (
            model_results["full"]
            [capacity]["hit_rate"]
        )

        print()
        print(
            f"CACHE CAPACITY: {capacity}"
        )
        print("-" * 78)

        print(
            f"LRU:       {lru_rate:>6.2%}"
        )

        print(
            f"Markov:    {markov_rate:>6.2%}"
        )

        print(
            f"Oracle:    {oracle_rate:>6.2%}"
        )

        print()

        ranked = []

        for model_name, results in model_results.items():

            rate = (
                results[capacity]["hit_rate"]
            )

            versus_markov = (
                rate - markov_rate
            )

            versus_full = (
                rate - full_rate
            )

            ranked.append(
                (
                    model_name,
                    rate,
                    versus_markov,
                    versus_full,
                )
            )

        ranked.sort(
            key=lambda item: item[1],
            reverse=True,
        )

        for (
            model_name,
            rate,
            versus_markov,
            versus_full,
        ) in ranked:

            print(
                f"{model_name:<24} "
                f"hit_rate={rate:>6.2%}  "
                f"vs_markov={versus_markov:+.2%}  "
                f"vs_full={versus_full:+.2%}"
            )


if __name__ == "__main__":
    main()
