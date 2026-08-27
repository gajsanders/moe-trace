from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass
class OverlapResult:
    layer: int
    comparisons: int
    mean_overlap: float
    min_overlap: float
    max_overlap: float


def expert_overlap(experts_a: list[int], experts_b: list[int]) -> float:
    """
    Fraction of selected experts shared by two routing decisions.

    Example:
        [1, 2, 3, 4] vs [2, 3, 5, 6]
        => 2 / 4 = 0.5
    """
    set_a = set(experts_a)
    set_b = set(experts_b)

    if not set_a:
        return 0.0

    return len(set_a & set_b) / len(set_a)


def calculate_layer_overlaps(events: Iterable[dict]) -> list[OverlapResult]:
    """
    Calculate adjacent-token expert overlap independently for each layer.
    """

    by_layer: dict[int, list[dict]] = defaultdict(list)

    for event in events:
        by_layer[event["layer"]].append(event)

    results = []

    for layer, layer_events in sorted(by_layer.items()):
        layer_events = sorted(
            layer_events,
            key=lambda event: event["token_index"],
        )

        overlaps = []

        for current, following in zip(
            layer_events,
            layer_events[1:],
        ):
            # Only compare genuinely adjacent tokens.
            if following["token_index"] != current["token_index"] + 1:
                continue

            overlaps.append(
                expert_overlap(
                    current["expert_ids"],
                    following["expert_ids"],
                )
            )

        if not overlaps:
            continue

        results.append(
            OverlapResult(
                layer=layer,
                comparisons=len(overlaps),
                mean_overlap=sum(overlaps) / len(overlaps),
                min_overlap=min(overlaps),
                max_overlap=max(overlaps),
            )
        )

    return results

