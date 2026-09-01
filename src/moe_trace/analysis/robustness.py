from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from statistics import mean, stdev


@dataclass
class PromptStats:
    prompt_id: str
    workload: str
    decode_tokens: int
    mean_decode_overlap: float
    mean_normalized_entropy: float
    mean_top16_share: float


@dataclass
class WorkloadStats:
    workload: str
    prompts: int
    mean_decode_overlap: float
    overlap_std: float
    mean_normalized_entropy: float
    mean_top16_share: float


def _overlap(a: list[int], b: list[int]) -> float:
    set_a = set(a)
    set_b = set(b)

    if not set_a:
        return 0.0

    return len(set_a & set_b) / len(set_a)


def _normalized_entropy(
    counts: Counter[int],
    num_experts: int = 128,
) -> float:
    total = sum(counts.values())

    if total == 0:
        return 0.0

    entropy = 0.0

    for count in counts.values():
        probability = count / total
        entropy -= probability * math.log2(probability)

    maximum_entropy = math.log2(num_experts)

    return entropy / maximum_entropy


def calculate_prompt_stats(
    events: list[dict],
    num_experts: int = 128,
) -> list[PromptStats]:

    by_prompt: dict[str, list[dict]] = defaultdict(list)

    for event in events:
        by_prompt[event["prompt_id"]].append(event)

    results = []

    for prompt_id, prompt_events in sorted(by_prompt.items()):

        workload = prompt_events[0]["workload"]

        decode_events = [
            event
            for event in prompt_events
            if event["phase"] == "decode"
        ]

        decode_tokens = len(
            {
                event["token_index"]
                for event in decode_events
            }
        )

        # ---------------------------------------------
        # Decode overlap
        # ---------------------------------------------

        by_layer: dict[int, list[dict]] = defaultdict(list)

        for event in decode_events:
            by_layer[event["layer"]].append(event)

        overlaps = []

        for layer_events in by_layer.values():

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

                overlaps.append(
                    _overlap(
                        current["expert_ids"],
                        following["expert_ids"],
                    )
                )

        # ---------------------------------------------
        # Layer concentration
        # ---------------------------------------------

        entropies = []
        top16_shares = []

        for layer_events in by_layer.values():

            counts: Counter[int] = Counter()

            for event in layer_events:
                counts.update(event["expert_ids"])

            total = sum(counts.values())

            entropies.append(
                _normalized_entropy(
                    counts,
                    num_experts=num_experts,
                )
            )

            top16 = sum(
                count
                for _, count
                in counts.most_common(16)
            )

            top16_shares.append(top16 / total)

        results.append(
            PromptStats(
                prompt_id=prompt_id,
                workload=workload,
                decode_tokens=decode_tokens,
                mean_decode_overlap=mean(overlaps),
                mean_normalized_entropy=mean(entropies),
                mean_top16_share=mean(top16_shares),
            )
        )

    return results


def calculate_workload_stats(
    prompt_stats: list[PromptStats],
) -> list[WorkloadStats]:

    by_workload: dict[
        str,
        list[PromptStats]
    ] = defaultdict(list)

    for stat in prompt_stats:
        by_workload[stat.workload].append(stat)

    results = []

    for workload, stats in sorted(by_workload.items()):

        overlaps = [
            stat.mean_decode_overlap
            for stat in stats
        ]

        results.append(
            WorkloadStats(
                workload=workload,
                prompts=len(stats),
                mean_decode_overlap=mean(overlaps),
                overlap_std=(
                    stdev(overlaps)
                    if len(overlaps) > 1
                    else 0.0
                ),
                mean_normalized_entropy=mean(
                    stat.mean_normalized_entropy
                    for stat in stats
                ),
                mean_top16_share=mean(
                    stat.mean_top16_share
                    for stat in stats
                ),
            )
        )

    return results


def calculate_layer_overlap_profiles(
    events: list[dict],
) -> dict[str, dict[int, float]]:
    """
    Return mean decode overlap for each layer in each prompt.
    """

    by_prompt_layer: dict[
        tuple[str, int],
        list[dict]
    ] = defaultdict(list)

    for event in events:
        if event["phase"] != "decode":
            continue

        by_prompt_layer[
            (event["prompt_id"], event["layer"])
        ].append(event)

    profiles: dict[str, dict[int, float]] = defaultdict(dict)

    for (prompt_id, layer), layer_events in by_prompt_layer.items():

        layer_events = sorted(
            layer_events,
            key=lambda event: event["token_index"],
        )

        overlaps = []

        for current, following in zip(
            layer_events,
            layer_events[1:],
        ):
            if (
                following["token_index"]
                != current["token_index"] + 1
            ):
                continue

            overlaps.append(
                _overlap(
                    current["expert_ids"],
                    following["expert_ids"],
                )
            )

        if overlaps:
            profiles[prompt_id][layer] = mean(overlaps)

    return dict(profiles)
