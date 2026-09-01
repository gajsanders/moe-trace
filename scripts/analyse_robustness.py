import json
from pathlib import Path
from statistics import mean

from moe_trace.analysis.robustness import (
    calculate_layer_overlap_profiles,
    calculate_prompt_stats,
    calculate_workload_stats,
)


TRACE_PATH = Path(
    "results/workload_routing_trace.jsonl"
)


def main():

    with TRACE_PATH.open() as f:
        events = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    prompt_stats = calculate_prompt_stats(events)
    workload_stats = calculate_workload_stats(prompt_stats)

    print("\nMoE Trace — Cross-Workload Robustness")
    print("=" * 72)

    # --------------------------------------------------
    # Prompt results
    # --------------------------------------------------

    print("\nPER-PROMPT DECODE RESULTS")
    print("-" * 72)

    for stat in prompt_stats:
        print(
            f"{stat.prompt_id:<12} "
            f"{stat.workload:<18} "
            f"overlap={stat.mean_decode_overlap:>6.1%}  "
            f"entropy={stat.mean_normalized_entropy:>6.1%}  "
            f"top16={stat.mean_top16_share:>6.1%}"
        )

    # --------------------------------------------------
    # Workload results
    # --------------------------------------------------

    print("\nWORKLOAD SUMMARY")
    print("-" * 72)

    for stat in workload_stats:
        print(
            f"{stat.workload:<18} "
            f"overlap={stat.mean_decode_overlap:>6.1%} "
            f"± {stat.overlap_std:>5.1%}  "
            f"entropy={stat.mean_normalized_entropy:>6.1%}  "
            f"top16={stat.mean_top16_share:>6.1%}"
        )

    # --------------------------------------------------
    # Overall results
    # --------------------------------------------------

    overall_overlap = mean(
        stat.mean_decode_overlap
        for stat in prompt_stats
    )

    overall_entropy = mean(
        stat.mean_normalized_entropy
        for stat in prompt_stats
    )

    overall_top16 = mean(
        stat.mean_top16_share
        for stat in prompt_stats
    )

    min_prompt = min(
        prompt_stats,
        key=lambda stat: stat.mean_decode_overlap,
    )

    max_prompt = max(
        prompt_stats,
        key=lambda stat: stat.mean_decode_overlap,
    )

    print("\nOVERALL")
    print("-" * 72)

    print(
        f"Mean decode overlap:      {overall_overlap:.1%}"
    )

    print(
        f"Mean normalized entropy: {overall_entropy:.1%}"
    )

    print(
        f"Mean top-16 share:       {overall_top16:.1%}"
    )

    print(
        f"Lowest-overlap prompt:   "
        f"{min_prompt.prompt_id} "
        f"({min_prompt.mean_decode_overlap:.1%})"
    )

    print(
        f"Highest-overlap prompt:  "
        f"{max_prompt.prompt_id} "
        f"({max_prompt.mean_decode_overlap:.1%})"
    )

    # --------------------------------------------------
    # Layer robustness
    # --------------------------------------------------

    profiles = calculate_layer_overlap_profiles(events)

    layer_means = {}

    for layer in range(48):

        values = [
            profile[layer]
            for profile in profiles.values()
            if layer in profile
        ]

        if values:
            layer_means[layer] = mean(values)

    print("\nMEAN DECODE OVERLAP BY LAYER")
    print("-" * 72)

    for layer, value in sorted(layer_means.items()):
        print(
            f"Layer {layer:>2}: {value:>6.1%}"
        )

    highest_layer = max(
        layer_means,
        key=layer_means.get,
    )

    lowest_layer = min(
        layer_means,
        key=layer_means.get,
    )

    print("\nLayer summary:")

    print(
        f"Highest mean overlap: "
        f"Layer {highest_layer} "
        f"({layer_means[highest_layer]:.1%})"
    )

    print(
        f"Lowest mean overlap:  "
        f"Layer {lowest_layer} "
        f"({layer_means[lowest_layer]:.1%})"
    )


if __name__ == "__main__":
    main()
