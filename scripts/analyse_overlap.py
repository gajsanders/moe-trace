import json
from pathlib import Path

from moe_trace.analysis.overlap import calculate_layer_overlaps


TRACE_PATH = Path("results/routing_trace.jsonl")


def main():
    with TRACE_PATH.open() as f:
        events = [json.loads(line) for line in f]

    results = calculate_layer_overlaps(events)

    if not results:
        raise RuntimeError("No adjacent-token comparisons found.")

    print("\nMoE Trace — Adjacent Token Expert Overlap")
    print("=" * 52)

    for result in results:
        print(
            f"Layer {result.layer:>2}: "
            f"mean={result.mean_overlap:>6.1%}  "
            f"min={result.min_overlap:>6.1%}  "
            f"max={result.max_overlap:>6.1%}  "
            f"n={result.comparisons}"
        )

    overall = sum(
        result.mean_overlap * result.comparisons
        for result in results
    ) / sum(result.comparisons for result in results)

    print("=" * 52)
    print(f"Overall adjacent-token overlap: {overall:.1%}")

    best = max(results, key=lambda r: r.mean_overlap)
    worst = min(results, key=lambda r: r.mean_overlap)

    print(
        f"Highest-overlap layer: {best.layer} "
        f"({best.mean_overlap:.1%})"
    )

    print(
        f"Lowest-overlap layer:  {worst.layer} "
        f"({worst.mean_overlap:.1%})"
    )


if __name__ == "__main__":
    main()
