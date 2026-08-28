import json
from pathlib import Path

from moe_trace.cache.simulator import simulate_policy
from moe_trace.cache.xgb_predictor import (
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
    32,
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

    train_prompts = all_prompts - TEST_PROMPTS

    print("\nMoE Trace — XGBoost Cache Experiment")
    print("=" * 72)

    print(f"Training prompts: {len(train_prompts)}")
    print(f"Test prompts:     {len(TEST_PROMPTS)}")

    print("\nHeld-out test prompts:")

    for prompt_id in sorted(TEST_PROMPTS):
        print(f"  {prompt_id}")

    print("\nBuilding leakage-safe training data...")

    X_train, y_train = build_training_data(
        events,
        train_prompt_ids=train_prompts,
    )

    print(
        f"Training rows: {len(y_train):,}"
    )

    print(
        f"Features:      {X_train.shape[1]}"
    )

    print(
        f"Positive rate: {y_train.mean():.1%}"
    )

    print("\nTraining XGBoost...")

    model = train_xgb_predictor(
        X_train,
        y_train,
    )

    print("Training complete.")

    # Only the held-out test prompts are used for all
    # policy comparisons below.
    test_events = [
        event
        for event in events
        if event["prompt_id"] in TEST_PROMPTS
    ]

    for capacity in CAPACITIES:

        print(
            f"\nCACHE CAPACITY: "
            f"{capacity} experts per layer"
        )

        print("-" * 72)

        lru = simulate_policy(
            test_events,
            policy="lru",
            capacity=capacity,
        )

        markov = simulate_policy(
            test_events,
            policy="markov",
            capacity=capacity,
        )

        oracle = simulate_policy(
            test_events,
            policy="oracle",
            capacity=capacity,
        )

        xgb_hits, xgb_misses = simulate_xgb_cache(
            events,
            test_prompt_ids=TEST_PROMPTS,
            model=model,
            capacity=capacity,
        )

        xgb_requests = xgb_hits + xgb_misses
        xgb_hit_rate = xgb_hits / xgb_requests

        print(
            f"lru        "
            f"hit_rate={lru.hit_rate:>6.1%}  "
            f"hits={lru.hits:>8}  "
            f"misses={lru.misses:>8}"
        )

        print(
            f"markov     "
            f"hit_rate={markov.hit_rate:>6.1%}  "
            f"hits={markov.hits:>8}  "
            f"misses={markov.misses:>8}"
        )

        print(
            f"xgboost    "
            f"hit_rate={xgb_hit_rate:>6.1%}  "
            f"hits={xgb_hits:>8}  "
            f"misses={xgb_misses:>8}"
        )

        print(
            f"oracle     "
            f"hit_rate={oracle.hit_rate:>6.1%}  "
            f"hits={oracle.hits:>8}  "
            f"misses={oracle.misses:>8}"
        )

        available_gap = (
            oracle.hit_rate - lru.hit_rate
        )

        markov_gain = (
            markov.hit_rate - lru.hit_rate
        )

        xgb_gain = (
            xgb_hit_rate - lru.hit_rate
        )

        xgb_vs_markov = (
            xgb_hit_rate - markov.hit_rate
        )

        gap_captured = (
            xgb_gain / available_gap
            if available_gap > 0
            else 0.0
        )

        print()

        print(
            f"Markov gain over LRU:   "
            f"{markov_gain:+.1%}"
        )

        print(
            f"XGBoost gain over LRU:  "
            f"{xgb_gain:+.1%}"
        )

        print(
            f"XGBoost vs Markov:      "
            f"{xgb_vs_markov:+.1%}"
        )

        print(
            f"Oracle headroom:        "
            f"{available_gap:.1%}"
        )

        print(
            f"Oracle gap captured:    "
            f"{gap_captured:.1%}"
        )

        print(
            f"LRU misses avoided:     "
            f"{lru.misses - xgb_misses:,}"
        )


if __name__ == "__main__":
    main()
