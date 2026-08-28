from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np
from xgboost import XGBClassifier


NUM_EXPERTS = 128
NUM_LAYERS = 48

WORKLOADS = {
    "coding": 0,
    "general_knowledge": 1,
    "math_reasoning": 2,
    "planning": 3,
    "summarisation": 4,
}

FEATURE_NAMES = [
    "expert_id",
    "layer",
    "current_membership",
    "current_router_score",
    "previous_membership",
    "previous_router_score",
    "recent_4",
    "recent_8",
    "age_since_last_use",
    "historical_frequency",
    "markov_score",
    "workload_coding",
    "workload_general",
    "workload_math",
    "workload_planning",
    "workload_summarisation",
]


def _score_map(event: dict) -> dict[int, float]:
    return dict(
        zip(
            event["expert_ids"],
            event["scores"],
        )
    )


def _features_for_expert(
    *,
    expert: int,
    layer: int,
    workload: str,
    current: dict,
    previous: dict | None,
    history: deque[set[int]],
    frequency: Counter[int],
    last_seen: dict[int, int],
    transition_counts: dict[int, Counter[int]],
    step: int,
) -> list[float]:

    current_set = set(current["expert_ids"])
    current_scores = _score_map(current)

    if previous is None:
        previous_set = set()
        previous_scores = {}
    else:
        previous_set = set(previous["expert_ids"])
        previous_scores = _score_map(previous)

    recent_4 = sum(
        expert in routing_set
        for routing_set in list(history)[-4:]
    )

    recent_8 = sum(
        expert in routing_set
        for routing_set in list(history)[-8:]
    )

    if expert in last_seen:
        age = step - last_seen[expert]
    else:
        age = 32

    age = min(age, 32)

    historical_frequency = (
        frequency[expert] / max(step + 1, 1)
    )

    markov_score = sum(
        transition_counts[source][expert]
        for source in current_set
    )

    workload_id = WORKLOADS[workload]

    workload_features = [
        1.0 if workload_id == index else 0.0
        for index in range(len(WORKLOADS))
    ]

    return [
        expert / (NUM_EXPERTS - 1),
        layer / (NUM_LAYERS - 1),
        1.0 if expert in current_set else 0.0,
        current_scores.get(expert, 0.0),
        1.0 if expert in previous_set else 0.0,
        previous_scores.get(expert, 0.0),
        recent_4 / 4.0,
        recent_8 / 8.0,
        age / 32.0,
        historical_frequency,
        np.log1p(markov_score),
        *workload_features,
    ]


def build_training_data(
    events: list[dict],
    train_prompt_ids: set[str],
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:

    rng = random.Random(seed)

    grouped: dict[
        tuple[str, int],
        list[dict],
    ] = defaultdict(list)

    for event in events:
        if event["phase"] != "decode":
            continue

        if event["prompt_id"] not in train_prompt_ids:
            continue

        grouped[
            (event["prompt_id"], event["layer"])
        ].append(event)

    rows = []
    labels = []

    for (_, layer), sequence in sorted(grouped.items()):

        sequence = sorted(
            sequence,
            key=lambda event: event["token_index"],
        )

        workload = sequence[0]["workload"]

        frequency: Counter[int] = Counter()
        last_seen: dict[int, int] = {}

        transition_counts: dict[
            int,
            Counter[int],
        ] = defaultdict(Counter)

        history: deque[set[int]] = deque(maxlen=8)

        previous = None

        for step in range(len(sequence) - 1):

            current = sequence[step]
            following = sequence[step + 1]

            current_set = set(current["expert_ids"])
            next_set = set(following["expert_ids"])

            # Learn only transitions that are already known.
            if previous is not None:
                previous_set = set(previous["expert_ids"])

                for source in previous_set:
                    for destination in current_set:
                        transition_counts[source][destination] += 1

            for expert in current_set:
                frequency[expert] += 1
                last_seen[expert] = step

            history.append(current_set)

            # All eight positive experts.
            positive_experts = sorted(next_set)

            # Sample an equal number of negative experts.
            negative_pool = sorted(
                set(range(NUM_EXPERTS)) - next_set
            )

            negative_experts = rng.sample(
                negative_pool,
                len(positive_experts),
            )

            candidates = [
                (expert, 1)
                for expert in positive_experts
            ] + [
                (expert, 0)
                for expert in negative_experts
            ]

            for expert, label in candidates:
                rows.append(
                    _features_for_expert(
                        expert=expert,
                        layer=layer,
                        workload=workload,
                        current=current,
                        previous=previous,
                        history=history,
                        frequency=frequency,
                        last_seen=last_seen,
                        transition_counts=transition_counts,
                        step=step,
                    )
                )

                labels.append(label)

            previous = current

    return (
        np.asarray(rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
    )


def train_xgb_predictor(
    X: np.ndarray,
    y: np.ndarray,
) -> XGBClassifier:

    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=4,
        random_state=42,
    )

    model.fit(X, y)

    return model


def simulate_xgb_cache(
    events: list[dict],
    test_prompt_ids: set[str],
    model: XGBClassifier,
    capacity: int,
    feature_indices: list[int] | None = None,
) -> tuple[int, int]:

    grouped: dict[
        tuple[str, int],
        list[dict],
    ] = defaultdict(list)

    for event in events:
        if event["phase"] != "decode":
            continue

        if event["prompt_id"] not in test_prompt_ids:
            continue

        grouped[
            (event["prompt_id"], event["layer"])
        ].append(event)

    hits = 0
    misses = 0

    for (_, layer), sequence in sorted(grouped.items()):

        sequence = sorted(
            sequence,
            key=lambda event: event["token_index"],
        )

        workload = sequence[0]["workload"]

        cache: set[int] = set()

        frequency: Counter[int] = Counter()
        last_seen: dict[int, int] = {}

        transition_counts: dict[
            int,
            Counter[int],
        ] = defaultdict(Counter)

        history: deque[set[int]] = deque(maxlen=8)

        previous = None

        for step, current in enumerate(sequence):

            requested = set(current["expert_ids"])

            hits += len(requested & cache)
            misses += len(requested - cache)

            if previous is not None:
                previous_set = set(previous["expert_ids"])

                for source in previous_set:
                    for destination in requested:
                        transition_counts[source][destination] += 1

            for expert in requested:
                frequency[expert] += 1
                last_seen[expert] = step

            history.append(requested)

            candidates = cache | requested

            if len(candidates) <= capacity:
                cache = candidates

            else:
                feature_rows = [
                    _features_for_expert(
                        expert=expert,
                        layer=layer,
                        workload=workload,
                        current=current,
                        previous=previous,
                        history=history,
                        frequency=frequency,
                        last_seen=last_seen,
                        transition_counts=transition_counts,
                        step=step,
                    )
                    for expert in sorted(candidates)
                ]

                if feature_indices is not None:
                    feature_rows = [
                        [
                            row[index]
                            for index in feature_indices
                        ]
                        for row in feature_rows
                    ]

                probabilities = model.predict_proba(
                    np.asarray(
                        feature_rows,
                        dtype=np.float32,
                    )
                )[:, 1]

                ranked = sorted(
                    zip(
                        sorted(candidates),
                        probabilities,
                    ),
                    key=lambda item: (
                        item[1],
                        last_seen.get(item[0], -1),
                        item[0],
                    ),
                    reverse=True,
                )

                cache = {
                    expert
                    for expert, _ in ranked[:capacity]
                }

            previous = current

    return hits, misses