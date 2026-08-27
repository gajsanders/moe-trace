from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass
class LayerRoutingStats:
    layer: int
    total_selections: int
    unique_experts: int
    entropy_bits: float
    normalized_entropy: float
    top_expert: int
    top_expert_share: float
    top_8_share: float
    top_16_share: float


@dataclass
class PhaseOverlapStats:
    phase: str
    comparisons: int
    mean_overlap: float


def shannon_entropy(counts: Counter[int]) -> float:
    total = sum(counts.values())

    if total == 0:
        return 0.0

    entropy = 0.0

    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)

    return entropy


def expert_frequency(events: Iterable[dict]) -> Counter[int]:
    counts: Counter[int] = Counter()

    for event in events:
        counts.update(event["expert_ids"])

    return counts


def calculate_layer_routing_stats(
    events: Iterable[dict],
    num_experts: int = 128,
) -> list[LayerRoutingStats]:

    by_layer: dict[int, list[dict]] = defaultdict(list)

    for event in events:
        by_layer[event["layer"]].append(event)

    results = []

    maximum_entropy = math.log2(num_experts)

    for layer, layer_events in sorted(by_layer.items()):
        counts: Counter[int] = Counter()

        for event in layer_events:
            counts.update(event["expert_ids"])

        total = sum(counts.values())

        entropy = shannon_entropy(counts)

        normalized_entropy = (
            entropy / maximum_entropy
            if maximum_entropy > 0
            else 0.0
        )

        most_common = counts.most_common()

        top_expert, top_count = most_common[0]

        top_8_count = sum(
            count for _, count in most_common[:8]
        )

        top_16_count = sum(
            count for _, count in most_common[:16]
        )

        results.append(
            LayerRoutingStats(
                layer=layer,
                total_selections=total,
                unique_experts=len(counts),
                entropy_bits=entropy,
                normalized_entropy=normalized_entropy,
                top_expert=top_expert,
                top_expert_share=top_count / total,
                top_8_share=top_8_count / total,
                top_16_share=top_16_count / total,
            )
        )

    return results


def calculate_phase_overlaps(
    events: Iterable[dict],
) -> list[PhaseOverlapStats]:

    by_phase_layer: dict[
        tuple[str, int],
        list[dict]
    ] = defaultdict(list)

    for event in events:
        by_phase_layer[
            (event["phase"], event["layer"])
        ].append(event)

    overlaps_by_phase: dict[str, list[float]] = defaultdict(list)

    for (phase, _layer), layer_events in by_phase_layer.items():
        layer_events = sorted(
            layer_events,
            key=lambda event: event["token_index"],
        )

        for current, following in zip(
            layer_events,
            layer_events[1:],
        ):
            if (
                following["token_index"]
                != current["token_index"] + 1
            ):
                continue

            current_experts = set(current["expert_ids"])
            following_experts = set(following["expert_ids"])

            overlap = (
                len(current_experts & following_experts)
                / len(current_experts)
            )

            overlaps_by_phase[phase].append(overlap)

    results = []

    for phase, overlaps in sorted(overlaps_by_phase.items()):
        if not overlaps:
            continue

        results.append(
            PhaseOverlapStats(
                phase=phase,
                comparisons=len(overlaps),
                mean_overlap=sum(overlaps) / len(overlaps),
            )
        )

    return results
