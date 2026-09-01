from __future__ import annotations

import json
from pathlib import Path

from moe_trace.cache.xgb_predictor import (
    FEATURE_NAMES,
    build_training_data,
    train_xgb_predictor,
)


TRACE_PATH = Path(
    "results/workload_routing_trace.jsonl"
)

OUTPUT_DIR = Path(
    "results/runtime_predictor"
)

TEST_PROMPTS = {
    "coding_05",
    "general_05",
    "math_05",
    "planning_05",
    "summary_05",
}

FEATURES = [
    "markov_score",
    "current_router_score",
    "previous_router_score",
]


def main() -> None:
    with TRACE_PATH.open() as handle:
        events = [
            json.loads(line)
            for line in handle
            if line.strip()
        ]

    all_prompts = {
        event["prompt_id"]
        for event in events
    }

    train_prompts = (
        all_prompts - TEST_PROMPTS
    )

    indices = [
        FEATURE_NAMES.index(name)
        for name in FEATURES
    ]

    print()
    print(
        "MoE Trace — Runtime Markov + Router Predictor"
    )
    print("=" * 72)
    print(
        f"Training prompts: {len(train_prompts)}"
    )
    print(
        f"Held-out prompts: {len(TEST_PROMPTS)}"
    )
    print(
        "Features: "
        + ", ".join(FEATURES)
    )

    print()
    print("Building training data...")

    X_full, y = build_training_data(
        events,
        train_prompt_ids=train_prompts,
    )

    X = X_full[:, indices]

    print(
        f"Rows:     {len(y):,}"
    )
    print(
        f"Features: {X.shape[1]}"
    )

    print()
    print("Training...")

    model = train_xgb_predictor(
        X,
        y,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        OUTPUT_DIR
        / "markov_router.json"
    )

    metadata_path = (
        OUTPUT_DIR
        / "markov_router_metadata.json"
    )

    model.save_model(
        model_path
    )

    metadata = {
        "features": FEATURES,
        "feature_indices": indices,
        "training_prompts": sorted(
            train_prompts
        ),
        "held_out_prompts": sorted(
            TEST_PROMPTS
        ),
        "training_rows": int(
            len(y)
        ),
        "model": {
            "n_estimators": 150,
            "max_depth": 6,
            "learning_rate": 0.08,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "tree_method": "hist",
            "n_jobs": 4,
            "random_state": 42,
        },
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        )
    )

    print()
    print(
        f"Model:    {model_path}"
    )
    print(
        f"Metadata: {metadata_path}"
    )


if __name__ == "__main__":
    main()
