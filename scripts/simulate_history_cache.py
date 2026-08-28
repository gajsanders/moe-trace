import json
from pathlib import Path

from moe_trace.cache.simulator import simulate_policy


TRACE_PATH = Path(
    "results/workload_routing_trace.jsonl"
)

CAPACITIES = [
    8,
    16,
    32,
]

POLICIES = [
    "lru",
    "markov",
    "history_markov",
    "oracle",
]


def main():

    with TRACE_PATH.open() as f:
        events = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    print("\nMoE Trace — Short-History Cache Experiment")
    print("=" * 72)

    for capacity in CAPACITIES:

        print(
            f"\nCACHE CAPACITY: "
            f"{capacity} experts per layer"
        )

        print("-" * 72)

        results = {}

        for policy in POLICIES:

            result = simulate_policy(
                events,
                policy=policy,
                capacity=capacity,
            )

            results[policy] = result

            print(
                f"{policy:<16} "
                f"hit_rate={result.hit_rate:>6.1%}  "
                f"hits={result.hits:>8}  "
                f"misses={result.misses:>8}"
            )

        lru = results["lru"]
        markov = results["markov"]
        history = results["history_markov"]
        oracle = results["oracle"]

        available_gap = (
            oracle.hit_rate - lru.hit_rate
        )

        history_gain = (
            history.hit_rate - lru.hit_rate
        )

        markov_gain = (
            markov.hit_rate - lru.hit_rate
        )

        extra_gain_over_markov = (
            history.hit_rate - markov.hit_rate
        )

        gap_captured = (
            history_gain / available_gap
            if available_gap > 0
            else 0.0
        )

        misses_avoided = (
            lru.misses - history.misses
        )

        print()
        print(
            f"Markov gain over LRU:        "
            f"{markov_gain:+.1%}"
        )

        print(
            f"History gain over LRU:       "
            f"{history_gain:+.1%}"
        )

        print(
            f"Extra gain over Markov:      "
            f"{extra_gain_over_markov:+.1%}"
        )

        print(
            f"Oracle headroom:             "
            f"{available_gap:.1%}"
        )

        print(
            f"Oracle gap captured:         "
            f"{gap_captured:.1%}"
        )

        print(
            f"LRU misses avoided:          "
            f"{misses_avoided:,}"
        )


if __name__ == "__main__":
    main()
