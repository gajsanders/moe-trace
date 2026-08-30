from __future__ import annotations

import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import mlx.core as mx
from mlx_lm import load


MODEL = "mlx-community/Qwen3-30B-A3B-4bit"
TRACE_PATH = Path("results/workload_routing_trace.jsonl")

LAYERS = [0, 7, 18, 31, 47]

ROUNDS = 8
SEED = 42


def load_sequences():
    grouped = defaultdict(list)

    with TRACE_PATH.open() as handle:
        for line in handle:
            event = json.loads(line)

            if event["phase"] != "decode":
                continue

            if event["layer"] not in LAYERS:
                continue

            grouped[
                (
                    event["prompt_id"],
                    event["layer"],
                )
            ].append(event)

    sequences = {}

    for key, events in grouped.items():
        sequences[key] = sorted(
            events,
            key=lambda event: event["token_index"],
        )

    return sequences


def make_indices(experts):
    return mx.array(
        experts,
        dtype=mx.int32,
    ).reshape(1, 1, 8)


def warm_layer(switch_mlp, x):
    """
    Touch all experts before measurement.

    This reduces first-use effects.
    """

    for start in range(0, 128, 8):
        experts = list(
            range(start, start + 8)
        )

        y = switch_mlp(
            x,
            make_indices(experts),
        )

        mx.eval(y)


def touch_projection(
    projection,
    expert_indices,
):
    """
    Read the complete selected expert parameter slices.

    The reduction forces MLX to read the selected data
    without retaining a large copied tensor as output.
    """

    values = []

    for parameter_name in [
        "weight",
        "scales",
        "biases",
    ]:
        if parameter_name not in projection:
            continue

        parameter = projection[
            parameter_name
        ]

        selected = mx.take(
            parameter,
            expert_indices,
            axis=0,
        )

        # Convert before reduction because the packed
        # weight tensor uses uint32 storage.
        checksum = mx.sum(
            selected.astype(mx.float32)
        )

        values.append(checksum)

    if values:
        mx.eval(*values)


def prefetch_experts(
    switch_mlp,
    experts,
):
    """
    Oracle pre-touch of exactly the experts required
    by the following SwitchGLU call.
    """

    indices = mx.array(
        experts,
        dtype=mx.int32,
    )

    touch_projection(
        switch_mlp.gate_proj,
        indices,
    )

    touch_projection(
        switch_mlp.up_proj,
        indices,
    )

    touch_projection(
        switch_mlp.down_proj,
        indices,
    )


def run_baseline(
    switch_mlp,
    x,
    events,
):
    compute_latencies = []

    for event in events:
        experts = event[
            "expert_ids"
        ]

        indices = make_indices(
            experts
        )

        start = time.perf_counter()

        y = switch_mlp(
            x,
            indices,
        )

        mx.eval(y)

        elapsed = (
            time.perf_counter()
            - start
        ) * 1000.0

        compute_latencies.append(
            elapsed
        )

    return compute_latencies


def run_prefetch(
    switch_mlp,
    x,
    events,
):
    prefetch_latencies = []
    compute_latencies = []
    total_latencies = []

    for event in events:
        experts = event[
            "expert_ids"
        ]

        indices = make_indices(
            experts
        )

        total_start = (
            time.perf_counter()
        )

        prefetch_start = (
            time.perf_counter()
        )

        prefetch_experts(
            switch_mlp,
            experts,
        )

        prefetch_elapsed = (
            time.perf_counter()
            - prefetch_start
        ) * 1000.0

        compute_start = (
            time.perf_counter()
        )

        y = switch_mlp(
            x,
            indices,
        )

        mx.eval(y)

        compute_elapsed = (
            time.perf_counter()
            - compute_start
        ) * 1000.0

        total_elapsed = (
            time.perf_counter()
            - total_start
        ) * 1000.0

        prefetch_latencies.append(
            prefetch_elapsed
        )

        compute_latencies.append(
            compute_elapsed
        )

        total_latencies.append(
            total_elapsed
        )

    return (
        prefetch_latencies,
        compute_latencies,
        total_latencies,
    )


def main():
    print()
    print(
        "MoE Trace — Oracle Expert Prefetch Benchmark"
    )
    print("=" * 78)

    print(f"Model:  {MODEL}")
    print(f"Trace:  {TRACE_PATH}")
    print(
        "Layers: "
        + ", ".join(
            map(str, LAYERS)
        )
    )
    print(f"Rounds: {ROUNDS}")

    sequences = load_sequences()

    print(
        "Prompt-layer sequences: "
        f"{len(sequences)}"
    )

    print()
    print("Loading model...")

    model, _ = load(MODEL)

    print("Model loaded.")

    rng = random.Random(SEED)

    layer_results = {
        layer: {
            "baseline": [],
            "prefetch_compute": [],
            "prefetch_cost": [],
            "prefetch_total": [],
        }
        for layer in LAYERS
    }

    paired_results = []

    for layer in LAYERS:

        print()
        print("=" * 78)
        print(f"LAYER {layer}")
        print("=" * 78)

        switch_mlp = (
            model.model.layers[
                layer
            ].mlp.switch_mlp
        )

        hidden_size = (
            switch_mlp.gate_proj.input_dims
        )

        x = mx.random.normal(
            shape=(1, 1, hidden_size)
        ).astype(mx.bfloat16)

        mx.eval(x)

        warm_layer(
            switch_mlp,
            x,
        )

        layer_sequences = [
            (
                prompt_id,
                events,
            )
            for (
                prompt_id,
                sequence_layer,
            ), events in sequences.items()
            if sequence_layer == layer
        ]

        layer_sequences.sort()

        for prompt_id, events in layer_sequences:

            baseline_values = []
            prefetch_compute_values = []
            prefetch_cost_values = []
            prefetch_total_values = []

            for round_index in range(
                ROUNDS
            ):
                conditions = [
                    "baseline",
                    "prefetch",
                ]

                rng.shuffle(
                    conditions
                )

                for condition in conditions:

                    if condition == "baseline":

                        values = run_baseline(
                            switch_mlp,
                            x,
                            events,
                        )

                        baseline_values.extend(
                            values
                        )

                    else:

                        (
                            prefetch_cost,
                            compute,
                            total,
                        ) = run_prefetch(
                            switch_mlp,
                            x,
                            events,
                        )

                        prefetch_cost_values.extend(
                            prefetch_cost
                        )

                        prefetch_compute_values.extend(
                            compute
                        )

                        prefetch_total_values.extend(
                            total
                        )

            baseline_median = (
                statistics.median(
                    baseline_values
                )
            )

            compute_median = (
                statistics.median(
                    prefetch_compute_values
                )
            )

            prefetch_median = (
                statistics.median(
                    prefetch_cost_values
                )
            )

            total_median = (
                statistics.median(
                    prefetch_total_values
                )
            )

            compute_delta = (
                (
                    compute_median
                    - baseline_median
                )
                / baseline_median
                * 100.0
            )

            total_delta = (
                (
                    total_median
                    - baseline_median
                )
                / baseline_median
                * 100.0
            )

            layer_results[
                layer
            ]["baseline"].extend(
                baseline_values
            )

            layer_results[
                layer
            ]["prefetch_compute"].extend(
                prefetch_compute_values
            )

            layer_results[
                layer
            ]["prefetch_cost"].extend(
                prefetch_cost_values
            )

            layer_results[
                layer
            ]["prefetch_total"].extend(
                prefetch_total_values
            )

            paired_results.append(
                (
                    layer,
                    prompt_id,
                    compute_delta,
                    total_delta,
                )
            )

            print(
                f"{prompt_id:<18} "
                f"base={baseline_median:.3f} ms  "
                f"warm={compute_median:.3f} ms  "
                f"compute Δ={compute_delta:+.2f}%  "
                f"prefetch={prefetch_median:.3f} ms  "
                f"net Δ={total_delta:+.2f}%"
            )

    print()
    print("=" * 78)
    print("LAYER SUMMARY")
    print("=" * 78)

    compute_deltas = []
    total_deltas = []

    for layer in LAYERS:

        baseline = statistics.median(
            layer_results[
                layer
            ]["baseline"]
        )

        warm_compute = statistics.median(
            layer_results[
                layer
            ]["prefetch_compute"]
        )

        prefetch_cost = statistics.median(
            layer_results[
                layer
            ]["prefetch_cost"]
        )

        total = statistics.median(
            layer_results[
                layer
            ]["prefetch_total"]
        )

        compute_delta = (
            (warm_compute - baseline)
            / baseline
            * 100.0
        )

        total_delta = (
            (total - baseline)
            / baseline
            * 100.0
        )

        compute_deltas.append(
            compute_delta
        )

        total_deltas.append(
            total_delta
        )

        print(
            f"Layer {layer:>2}: "
            f"base={baseline:.3f} ms  "
            f"warm={warm_compute:.3f} ms  "
            f"compute Δ={compute_delta:+.2f}%  "
            f"prefetch={prefetch_cost:.3f} ms  "
            f"net Δ={total_delta:+.2f}%"
        )

    print()
    print("=" * 78)
    print("AGGREGATE")
    print("=" * 78)

    print(
        "Median warmed-compute delta: "
        f"{statistics.median(compute_deltas):+.2f}%"
    )

    print(
        "Median net prefetch delta:    "
        f"{statistics.median(total_deltas):+.2f}%"
    )

    warmed_faster = sum(
        delta < 0
        for delta in compute_deltas
    )

    net_faster = sum(
        delta < 0
        for delta in total_deltas
    )

    print(
        "Layers where warmed compute "
        "was faster: "
        f"{warmed_faster}/{len(LAYERS)}"
    )

    print(
        "Layers where total prefetch path "
        "was faster: "
        f"{net_faster}/{len(LAYERS)}"
    )

    print()
    print("=" * 78)
    print("DECISION RULE")
    print("=" * 78)

    print(
        "A negative warmed-compute delta means "
        "pre-touch reduced later SwitchGLU latency."
    )

    print(
        "A negative net delta means synchronous "
        "prefetch was profitable by itself."
    )

    print(
        "If warmed compute is not consistently faster, "
        "stop prefetch work."
    )

    print(
        "If warmed compute is faster but net time is slower, "
        "consider asynchronous overlap as the next experiment."
    )

    print(
        "If total time is already faster, "
        "move directly to end-to-end validation."
    )


if __name__ == "__main__":
    main()
