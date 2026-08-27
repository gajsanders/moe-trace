import json
from pathlib import Path

from moe_trace.analysis.routing_stats import (
    calculate_layer_routing_stats,
    calculate_phase_overlaps,
    expert_frequency,
)


TRACE_PATH = Path("results/routing_trace.jsonl")


def main():
    with TRACE_PATH.open() as f:
        events = [json.loads(line) for line in f]

    print("\nMoE Trace — Routing Characterisation")
    print("=" * 64)

    # ---------------------------------------------------------
    # Global expert frequency
    # ---------------------------------------------------------

    frequencies = expert_frequency(events)

    total_selections = sum(frequencies.values())

    print("\nGLOBAL EXPERT FREQUENCY")
    print("-" * 64)

    print(f"Total expert selections: {total_selections}")
    print(f"Unique experts observed: {len(frequencies)}")

    print("\nTop 15 experts:")

    for expert, count in frequencies.most_common(15):
        share = count / total_selections

        print(
            f"Expert {expert:>3}: "
            f"{count:>5} selections "
            f"({share:>6.2%})"
        )

    # ---------------------------------------------------------
    # Per-layer concentration
    # ---------------------------------------------------------

    layer_stats = calculate_layer_routing_stats(events)

    print("\nLAYER ROUTING CONCENTRATION")
    print("-" * 64)

    for stat in layer_stats:
        print(
            f"Layer {stat.layer:>2}: "
            f"unique={stat.unique_experts:>3}  "
            f"entropy={stat.normalized_entropy:>6.1%}  "
            f"top1={stat.top_expert_share:>6.1%}  "
            f"top8={stat.top_8_share:>6.1%}  "
            f"top16={stat.top_16_share:>6.1%}"
        )

    lowest_entropy = min(
        layer_stats,
        key=lambda stat: stat.normalized_entropy,
    )

    highest_entropy = max(
        layer_stats,
        key=lambda stat: stat.normalized_entropy,
    )

    print("\nLayer concentration summary:")

    print(
        f"Most concentrated layer: "
        f"{lowest_entropy.layer} "
        f"(normalized entropy "
        f"{lowest_entropy.normalized_entropy:.1%})"
    )

    print(
        f"Most diffuse layer: "
        f"{highest_entropy.layer} "
        f"(normalized entropy "
        f"{highest_entropy.normalized_entropy:.1%})"
    )

    # ---------------------------------------------------------
    # Prefill vs decode overlap
    # ---------------------------------------------------------

    phase_stats = calculate_phase_overlaps(events)

    print("\nPREFILL VS DECODE ADJACENT-TOKEN OVERLAP")
    print("-" * 64)

    for stat in phase_stats:
        print(
            f"{stat.phase:<7}: "
            f"mean overlap={stat.mean_overlap:>6.1%}  "
            f"comparisons={stat.comparisons}"
        )


if __name__ == "__main__":
    main()
