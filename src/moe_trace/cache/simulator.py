from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable


@dataclass
class CacheResult:
    policy: str
    capacity: int
    hits: int
    misses: int
    requests: int

    @property
    def hit_rate(self) -> float:
        if self.requests == 0:
            return 0.0

        return self.hits / self.requests


def _prepare_sequences(
    events: Iterable[dict],
) -> list[list[set[int]]]:
    """
    Build independent decode routing sequences.

    Each sequence is one prompt + one MoE layer.
    Each item in a sequence is the set of experts requested
    for one decode token.
    """

    grouped: dict[
        tuple[str, int],
        list[dict]
    ] = defaultdict(list)

    for event in events:
        if event["phase"] != "decode":
            continue

        grouped[
            (event["prompt_id"], event["layer"])
        ].append(event)

    sequences = []

    for _, group in sorted(grouped.items()):

        ordered = sorted(
            group,
            key=lambda event: event["token_index"],
        )

        sequences.append(
            [
                set(event["expert_ids"])
                for event in ordered
            ]
        )

    return sequences


def _simulate_lru(
    sequence: list[set[int]],
    capacity: int,
) -> tuple[int, int]:

    cache: set[int] = set()
    last_used: dict[int, int] = {}

    hits = 0
    misses = 0

    for step, requested in enumerate(sequence):

        hits += len(requested & cache)
        misses += len(requested - cache)

        for expert in requested:
            cache.add(expert)
            last_used[expert] = step

        if len(cache) > capacity:

            keep = sorted(
                cache,
                key=lambda expert: (
                    last_used.get(expert, -1),
                    expert,
                ),
                reverse=True,
            )[:capacity]

            cache = set(keep)

    return hits, misses


def _simulate_lfu(
    sequence: list[set[int]],
    capacity: int,
) -> tuple[int, int]:

    cache: set[int] = set()

    frequency: Counter[int] = Counter()
    last_used: dict[int, int] = {}

    hits = 0
    misses = 0

    for step, requested in enumerate(sequence):

        hits += len(requested & cache)
        misses += len(requested - cache)

        for expert in requested:
            frequency[expert] += 1
            last_used[expert] = step
            cache.add(expert)

        if len(cache) > capacity:

            keep = sorted(
                cache,
                key=lambda expert: (
                    frequency[expert],
                    last_used.get(expert, -1),
                    expert,
                ),
                reverse=True,
            )[:capacity]

            cache = set(keep)

    return hits, misses


def _simulate_random(
    sequence: list[set[int]],
    capacity: int,
    seed: int,
) -> tuple[int, int]:

    rng = random.Random(seed)

    cache: set[int] = set()

    hits = 0
    misses = 0

    for requested in sequence:

        hits += len(requested & cache)
        misses += len(requested - cache)

        cache.update(requested)

        if len(cache) > capacity:
            cache = set(
                rng.sample(
                    sorted(cache),
                    capacity,
                )
            )

    return hits, misses


def _simulate_oracle(
    sequence: list[set[int]],
    capacity: int,
) -> tuple[int, int]:
    """
    Future-aware cache.

    After each request, retain experts whose next use is nearest.

    This provides an approximate offline optimum for the
    event-level cache model.
    """

    cache: set[int] = set()

    hits = 0
    misses = 0

    for step, requested in enumerate(sequence):

        hits += len(requested & cache)
        misses += len(requested - cache)

        candidates = cache | requested

        if len(candidates) <= capacity:
            cache = candidates
            continue

        future = sequence[step + 1:]

        next_use = {}

        for expert in candidates:

            distance = float("inf")

            for offset, future_request in enumerate(
                future,
                start=1,
            ):
                if expert in future_request:
                    distance = offset
                    break

            next_use[expert] = distance

        keep = sorted(
            candidates,
            key=lambda expert: (
                next_use[expert],
                expert,
            ),
        )[:capacity]

        cache = set(keep)

    return hits, misses

def _simulate_markov(
    sequence: list[set[int]],
    capacity: int,
) -> tuple[int, int]:
    """
    Online first-order transition-aware cache.

    The policy learns:
        P(next expert | current expert)

    from routing transitions that have already occurred.

    It does not use future requests.
    """

    cache: set[int] = set()

    # transitions[a][b] counts how often expert b appeared
    # after expert a in an already-observed transition.
    transitions: dict[int, Counter[int]] = defaultdict(Counter)

    frequency: Counter[int] = Counter()
    last_used: dict[int, int] = {}

    hits = 0
    misses = 0

    previous_request: set[int] | None = None

    for step, requested in enumerate(sequence):

        hits += len(requested & cache)
        misses += len(requested - cache)

        # The transition previous -> current is now known.
        # Learn it only after current has actually arrived.
        if previous_request is not None:
            for previous_expert in previous_request:
                for current_expert in requested:
                    transitions[previous_expert][current_expert] += 1

        for expert in requested:
            frequency[expert] += 1
            last_used[expert] = step

        candidates = cache | requested

        if len(candidates) > capacity:

            # Predict what is likely after the CURRENT request.
            predictive_score = {}

            for candidate in candidates:
                score = 0

                for current_expert in requested:
                    score += transitions[
                        current_expert
                    ][candidate]

                predictive_score[candidate] = score

            keep = sorted(
                candidates,
                key=lambda expert: (
                    predictive_score[expert],
                    frequency[expert],
                    last_used.get(expert, -1),
                    expert,
                ),
                reverse=True,
            )[:capacity]

            cache = set(keep)

        else:
            cache = candidates

        previous_request = requested

    return hits, misses

def _simulate_runtime_lru(
    sequence: list[set[int]],
    capacity: int,
) -> tuple[int, int]:
    """
    Runtime-faithful token-level LRU approximation.

    All experts required by the current request are protected
    until that request has completed.

    When capacity is exceeded, eviction can remove only experts
    that are not part of the current request.
    """

    cache: set[int] = set()
    last_used: dict[int, int] = {}

    hits = 0
    misses = 0

    for step, requested in enumerate(sequence):
        hits += len(requested & cache)
        misses += len(requested - cache)

        for expert in requested:
            last_used[expert] = step

        candidates = cache | requested

        if len(candidates) > capacity:
            evictable = candidates - requested

            remove_count = (
                len(candidates) - capacity
            )

            if remove_count > len(evictable):
                raise ValueError(
                    "capacity is too small to protect "
                    "the current expert request"
                )

            victims = sorted(
                evictable,
                key=lambda expert: (
                    last_used.get(expert, -1),
                    expert,
                ),
            )[:remove_count]

            cache = (
                candidates - set(victims)
            )

        else:
            cache = candidates

    return hits, misses


def _simulate_runtime_markov(
    sequence: list[set[int]],
    capacity: int,
) -> tuple[int, int]:
    """
    Runtime-faithful causal Markov cache.

    The Markov prediction is identical in form to
    _simulate_markov(), but experts required by the current
    request cannot be eviction victims.

    This models a runtime where the persistent cache is also
    the physical working storage used by the current MoE call.
    """

    cache: set[int] = set()

    transitions: dict[
        int,
        Counter[int],
    ] = defaultdict(Counter)

    frequency: Counter[int] = Counter()
    last_used: dict[int, int] = {}

    hits = 0
    misses = 0

    previous_request: set[int] | None = None

    for step, requested in enumerate(sequence):
        hits += len(requested & cache)
        misses += len(requested - cache)

        # Learn previous -> current only after current
        # routing has actually been observed.
        if previous_request is not None:
            for previous_expert in previous_request:
                for current_expert in requested:
                    transitions[
                        previous_expert
                    ][current_expert] += 1

        for expert in requested:
            frequency[expert] += 1
            last_used[expert] = step

        candidates = cache | requested

        if len(candidates) > capacity:
            # Current experts must remain resident until the
            # current SwitchGLU operation has completed.
            evictable = candidates - requested

            remove_count = (
                len(candidates) - capacity
            )

            if remove_count > len(evictable):
                raise ValueError(
                    "capacity is too small to protect "
                    "the current expert request"
                )

            predictive_score: dict[
                int,
                int,
            ] = {}

            for candidate in evictable:
                score = 0

                for current_expert in requested:
                    score += transitions[
                        current_expert
                    ][candidate]

                predictive_score[
                    candidate
                ] = score

            # Match the original Markov simulator's
            # ranking exactly:
            #
            # predictive score
            # historical frequency
            # recency
            # expert id
            #
            # Lowest-ranked evictable experts are removed.
            victims = sorted(
                evictable,
                key=lambda expert: (
                    predictive_score[expert],
                    frequency[expert],
                    last_used.get(
                        expert,
                        -1,
                    ),
                    expert,
                ),
            )[:remove_count]

            cache = (
                candidates - set(victims)
            )

        else:
            cache = candidates

        previous_request = requested

    return hits, misses

def _simulate_history_markov(
    sequence: list[set[int]],
    capacity: int,
    history_length: int = 3,
    decay: float = 0.5,
) -> tuple[int, int]:
    """
    Online short-history transition-aware cache.

    Uses up to the last `history_length` routing sets.

    More recent routing sets receive larger weights.

    The policy learns transitions from routing events that
    have already occurred. It does not use future requests.
    """

    cache: set[int] = set()

    transitions: dict[int, Counter[int]] = defaultdict(Counter)

    frequency: Counter[int] = Counter()
    last_used: dict[int, int] = {}

    history: list[set[int]] = []

    hits = 0
    misses = 0

    previous_request: set[int] | None = None

    for step, requested in enumerate(sequence):

        hits += len(requested & cache)
        misses += len(requested - cache)

        # Learn the transition only after the current
        # routing request becomes known.
        if previous_request is not None:
            for previous_expert in previous_request:
                for current_expert in requested:
                    transitions[previous_expert][current_expert] += 1

        for expert in requested:
            frequency[expert] += 1
            last_used[expert] = step

        candidates = cache | requested

        # Add the current routing state to history.
        history.append(requested)

        if len(history) > history_length:
            history.pop(0)

        if len(candidates) > capacity:

            predictive_score = {}

            for candidate in candidates:
                score = 0.0

                # Most recent state gets weight 1.0.
                # Earlier states receive decayed weights.
                for age, routing_set in enumerate(
                    reversed(history)
                ):
                    weight = decay ** age

                    for source_expert in routing_set:
                        score += (
                            weight
                            * transitions[
                                source_expert
                            ][candidate]
                        )

                predictive_score[candidate] = score

            keep = sorted(
                candidates,
                key=lambda expert: (
                    predictive_score[expert],
                    frequency[expert],
                    last_used.get(expert, -1),
                    expert,
                ),
                reverse=True,
            )[:capacity]

            cache = set(keep)

        else:
            cache = candidates

        previous_request = requested

    return hits, misses


def simulate_policy(
    events: Iterable[dict],
    policy: str,
    capacity: int,
    seed: int = 42,
) -> CacheResult:

    if capacity < 8:
        raise ValueError(
            "Cache capacity must be at least 8 experts."
        )

    sequences = _prepare_sequences(events)

    total_hits = 0
    total_misses = 0

    for index, sequence in enumerate(sequences):

        if policy == "lru":
            hits, misses = _simulate_lru(
                sequence,
                capacity,
            )

        elif policy == "lfu":
            hits, misses = _simulate_lfu(
                sequence,
                capacity,
            )

        elif policy == "random":
            hits, misses = _simulate_random(
                sequence,
                capacity,
                seed + index,
            )

        elif policy == "markov":
            hits, misses = _simulate_markov(
            sequence,
            capacity,
            )

        elif policy == "runtime_lru":
            hits, misses = _simulate_runtime_lru(
                sequence,
                capacity,
            )

        elif policy == "runtime_markov":
            hits, misses = _simulate_runtime_markov(
            sequence,
            capacity,
            )

        elif policy == "history_markov":
            hits, misses = _simulate_history_markov(
            sequence,
            capacity,
            )

        elif policy == "oracle":
            hits, misses = _simulate_oracle(
                sequence,
                capacity,
            )

        else:
            raise ValueError(
                f"Unknown cache policy: {policy}"
            )

        total_hits += hits
        total_misses += misses

    return CacheResult(
        policy=policy,
        capacity=capacity,
        hits=total_hits,
        misses=total_misses,
        requests=total_hits + total_misses,
    )
