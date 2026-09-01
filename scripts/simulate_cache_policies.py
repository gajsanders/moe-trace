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
    64,
]

POLICIES = [
    "random",
    "lfu",
    "lru",
    "oracle",
]


def main():

    with TRACE_PATH.open() as f:
        events = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    print("\nMoE Trace — Cache Policy Simulation")
    print("=" * 72)

    results = []

    for capacity in CAPACITIES:

        print(
            f"\nCACHE CAPACITY: "
            f"{capacity} experts per layer"
        )

        print("-" * 72)

        capacity_results = {}

        for policy in POLICIES:

            result = simulate_policy(
                events,
                policy=policy,
                capacity=capacity,
            )

            capacity_results[policy] = result
            results.append(result)

            print(
                f"{policy:<10} "
                f"hit_rate={result.hit_rate:>6.1%}  "
                f"hits={result.hits:>8}  "
                f"misses={result.misses:>8}"
            )

        lru = capacity_results["lru"]
        oracle = capacity_results["oracle"]

        gap = oracle.hit_rate - lru.hit_rate

        avoidable_misses = (
            lru.misses - oracle.misses
        )

        print(
            f"\nLRU → Oracle gap: "
            f"{gap:.1%}"
        )

        print(
            f"Potential avoidable misses: "
            f"{avoidable_misses:,}"
        )


if __name__ == "__main__":
    main()
