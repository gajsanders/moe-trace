from __future__ import annotations

import contextlib
import random
import statistics
import time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate

from mlx_lm.models.qwen3_moe import (
    Qwen3MoeSparseMoeBlock,
)


MODEL = "mlx-community/Qwen3-30B-A3B-4bit"

PROMPT = (
    "/no_think Explain why mixture-of-experts models "
    "can be efficient. Give a concise technical answer."
)

GENERATED_TOKENS = 128
WARMUP_RUNS = 2
MEASURE_RUNS = 8

SEED = 42


@contextlib.contextmanager
def bypass_switchglu():
    """
    Keep the Qwen3 MoE router path intact.

    Replace only the selected-expert SwitchGLU computation
    with a zero tensor of the same output shape.

    This is a timing experiment only.
    Generated text is not meaningful in bypass mode.
    """

    original_call = (
        Qwen3MoeSparseMoeBlock.__call__
    )

    def bypass_call(self, x):
        # Preserve router calculation.
        gates = self.gate(x)

        gates = mx.softmax(
            gates,
            axis=-1,
            precise=True,
        )

        k = self.top_k

        inds = mx.argpartition(
            gates,
            kth=-k,
            axis=-1,
        )[..., -k:]

        inds = mx.stop_gradient(
            inds
        )

        scores = mx.take_along_axis(
            gates,
            inds,
            axis=-1,
        )

        # Preserve router calculation.
        gates = self.gate(x)

        gates = mx.softmax(
            gates,
            axis=-1,
            precise=True,
        )

        k = self.top_k

        inds = mx.argpartition(
            gates,
            kth=-k,
            axis=-1,
        )[..., -k:]

        inds = mx.stop_gradient(
            inds
        )

        scores = mx.take_along_axis(
            gates,
            inds,
            axis=-1,
        )

        if self.norm_topk_prob:
            scores /= mx.sum(
                scores,
                axis=-1,
                keepdims=True,
            )

        # Preserve the dependency on router outputs
        # so the router remains part of the graph.
        router_anchor = (
            mx.sum(
                scores.astype(
                    mx.float32
                ),
                axis=-1,
                keepdims=True,
            )
            * 0.0
        ).astype(x.dtype)

        # Same final MoE output shape as normal.
        return (
            mx.zeros_like(x)
            + router_anchor
        )

    Qwen3MoeSparseMoeBlock.__call__ = (
        bypass_call
    )

    try:
        yield
    finally:
        Qwen3MoeSparseMoeBlock.__call__ = (
            original_call
        )


def run_generation(
    model,
    tokenizer,
) -> tuple[float, float, int]:

    start = time.perf_counter()

    last_response = None

    for response in stream_generate(
        model,
        tokenizer,
        PROMPT,
        max_tokens=GENERATED_TOKENS,
    ):
        last_response = response

    elapsed = (
        time.perf_counter()
        - start
    )

    if last_response is None:
        raise RuntimeError(
            "Generation returned no response."
        )

    return (
        elapsed,
        last_response.generation_tps,
        last_response.generation_tokens,
    )


def measure_condition(
    *,
    model,
    tokenizer,
    bypass: bool,
) -> tuple[float, float]:

    context = (
        bypass_switchglu()
        if bypass
        else contextlib.nullcontext()
    )

    with context:
        elapsed, tps, tokens = (
            run_generation(
                model,
                tokenizer,
            )
        )

    ms_per_token = (
        elapsed
        / tokens
        * 1000.0
    )

    return (
        ms_per_token,
        tps,
    )


def main():
    print()
    print(
        "MoE Trace — End-to-End MoE Decode Share Benchmark"
    )
    print("=" * 78)

    print(f"Model:           {MODEL}")
    print(f"Generated:       {GENERATED_TOKENS} tokens")
    print(f"Warm-up runs:    {WARMUP_RUNS}")
    print(f"Measured runs:   {MEASURE_RUNS}")

    print()
    print("Loading model...")

    model, tokenizer = load(
        MODEL
    )

    print("Model loaded.")

    print()
    print(
        "Normal = unchanged Qwen3 execution."
    )

    print(
        "Bypass = router retained, "
        "SwitchGLU expert computation removed."
    )

    print()
    print("Warm-up")
    print("-" * 78)

    for index in range(
        WARMUP_RUNS
    ):
        normal_ms, _ = (
            measure_condition(
                model=model,
                tokenizer=tokenizer,
                bypass=False,
            )
        )

        bypass_ms, _ = (
            measure_condition(
                model=model,
                tokenizer=tokenizer,
                bypass=True,
            )
        )

        print(
            f"Warm-up {index + 1}: "
            f"normal={normal_ms:.3f} ms/token  "
            f"bypass={bypass_ms:.3f} ms/token"
        )

    normal_results = []
    bypass_results = []

    rng = random.Random(SEED)

    print()
    print("Measured runs")
    print("-" * 78)

    for run_index in range(
        MEASURE_RUNS
    ):

        conditions = [
            "normal",
            "bypass",
        ]

        rng.shuffle(
            conditions
        )

        run_values = {}

        for condition in conditions:

            # Reduce cache carry-over between runs.
            mx.clear_cache()

            is_bypass = (
                condition == "bypass"
            )

            ms_per_token, tps = (
                measure_condition(
                    model=model,
                    tokenizer=tokenizer,
                    bypass=is_bypass,
                )
            )

            run_values[
                condition
            ] = (
                ms_per_token,
                tps,
            )

            if is_bypass:
                bypass_results.append(
                    ms_per_token
                )
            else:
                normal_results.append(
                    ms_per_token
                )

        normal_ms, normal_tps = (
            run_values["normal"]
        )

        bypass_ms, bypass_tps = (
            run_values["bypass"]
        )

        removed_ms = (
            normal_ms
            - bypass_ms
        )

        share = (
            removed_ms
            / normal_ms
            * 100.0
        )

        print(
            f"Run {run_index + 1}: "
            f"normal={normal_ms:.3f} ms/token "
            f"({normal_tps:.1f} tok/s)  "
            f"bypass={bypass_ms:.3f} ms/token "
            f"({bypass_tps:.1f} tok/s)  "
            f"removed={removed_ms:.3f} ms  "
            f"share={share:.1f}%"
        )

    normal_median = (
        statistics.median(
            normal_results
        )
    )

    bypass_median = (
        statistics.median(
            bypass_results
        )
    )

    removed_median = (
        normal_median
        - bypass_median
    )

    estimated_share = (
        removed_median
        / normal_median
        * 100.0
    )

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)

    print(
        f"Median normal decode: "
        f"{normal_median:.3f} ms/token"
    )

    print(
        f"Median bypass decode: "
        f"{bypass_median:.3f} ms/token"
    )

    print(
        f"Estimated expert-path cost: "
        f"{removed_median:.3f} ms/token"
    )

    print(
        f"Estimated expert-path share: "
        f"{estimated_share:.1f}%"
    )

    print()
    print("=" * 78)
    print("LOCALITY UPSIDE ESTIMATE")
    print("=" * 78)

    # Finding 14A:
    # median layer-level natural-routing advantage
    # relative to shuffled routing.
    locality_effect = 0.0146

    estimated_current_benefit = (
        estimated_share
        / 100.0
        * locality_effect
        * 100.0
    )

    print(
        "Observed real-routing locality effect: "
        "1.46% of isolated SwitchGLU time"
    )

    print(
        "Approximate whole-token equivalent: "
        f"{estimated_current_benefit:.2f}%"
    )

    print()
    print(
        "This multiplication is an upper-level "
        "estimate, not a measured speedup."
    )

    print()
    print("=" * 78)
    print("INTERPRETATION LIMIT")
    print("=" * 78)

    print(
        "Bypass mode changes model activations "
        "and therefore changes later routing."
    )

    print(
        "The result is an A/B cost estimate, "
        "not exact kernel profiling."
    )

    print(
        "Bypass output is not semantically valid."
    )

    print(
        "Normal generation remains the "
        "end-to-end performance baseline."
    )


if __name__ == "__main__":
    main()
