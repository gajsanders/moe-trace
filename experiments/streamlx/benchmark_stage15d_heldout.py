from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

from mlx_lm.generate import stream_generate
from streamlx.integrate import aggregate_stats, load_streaming_model


MODEL_PATH = (
    "/Users/enceladus/.cache/huggingface/hub/"
    "models--mlx-community--Qwen3-30B-A3B-4bit/"
    "snapshots/d388dead1515f5e085ef7a0431dd8fadf0886c57"
)

PROMPT_FILE = Path(
    "/Users/enceladus/Documents/moe-trace/"
    "prompts/workload_suite.jsonl"
)

MAX_TOKENS = 64

SELECTED_IDS = {
    "coding_05",
    "general_05",
    "math_05",
    "planning_05",
    "summary_05",
}


def load_prompts() -> list[dict]:
    prompts = []

    with PROMPT_FILE.open() as handle:
        for line in handle:
            item = json.loads(line)

            prompt_id = item.get("prompt_id", item.get("id"))
            if prompt_id not in SELECTED_IDS:
                continue

            prompt_text = item.get("prompt", item.get("text"))
            if prompt_text is None:
                raise ValueError(f"No prompt text found for {prompt_id}")

            prompts.append(
                {
                    "prompt_id": prompt_id,
                    "workload": item.get("workload", "unknown"),
                    "prompt": prompt_text,
                }
            )

    prompts.sort(key=lambda item: item["prompt_id"])

    found = {item["prompt_id"] for item in prompts}
    missing = sorted(SELECTED_IDS - found)

    if missing:
        raise RuntimeError(f"Missing prompts: {missing}")

    return prompts


def reset_predictor_state(pools: dict) -> None:
    """Reset causal learning state at a prompt boundary.

    Resident cache contents and runtime counters remain intact.
    """
    for pool in pools.values():
        reset = getattr(pool, "reset_markov_state", None)
        if reset is not None:
            reset()


def stat_delta(before: dict, after: dict, key: str):
    return after.get(key, 0) - before.get(key, 0)


def run_prompt(model, tokenizer, prompt: str) -> dict:
    start = time.perf_counter()
    last_response = None

    for response in stream_generate(
        model,
        tokenizer,
        prompt,
        max_tokens=MAX_TOKENS,
    ):
        last_response = response

    elapsed = time.perf_counter() - start

    if last_response is None:
        raise RuntimeError("Generation returned no response.")

    tokens = last_response.generation_tokens

    return {
        "tokens": tokens,
        "elapsed_s": elapsed,
        "wall_tps": tokens / elapsed if elapsed > 0 else 0.0,
        "reported_tps": last_response.generation_tps,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-gib", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    policy = os.environ.get("STREAMLX_EVICT", "lru")
    prefetch = os.environ.get("STREAMLX_PREFETCH", "1") != "0"

    budget_bytes = int(args.budget_gib * (1 << 30))
    prompts = load_prompts()

    print()
    print("MoE Trace — Stage 15D Held-Out Runtime Benchmark")
    print("=" * 78)
    print(f"Policy:       {policy}")
    print(f"Budget:       {args.budget_gib:.1f} GiB")
    print(f"Prompts:      {len(prompts)} held-out")
    print(f"Max tokens:   {MAX_TOKENS}")
    print(f"Prefetch:     {'enabled' if prefetch else 'disabled'}")

    print()
    print("Loading model...")

    model, tokenizer, pools, reader = load_streaming_model(
        MODEL_PATH,
        budget_bytes=budget_bytes,
    )

    print("Model loaded.")

    initial_stats = aggregate_stats(pools)
    prompt_results = []

    for index, item in enumerate(prompts, start=1):
        reset_predictor_state(pools)

        before = aggregate_stats(pools)

        result = run_prompt(
            model,
            tokenizer,
            item["prompt"],
        )

        after = aggregate_stats(pools)

        hits = stat_delta(before, after, "hits")
        misses = stat_delta(before, after, "misses")
        evictions = stat_delta(before, after, "evictions")
        fetch_s = stat_delta(before, after, "fetch_s")
        predictor_calls = stat_delta(
            before,
            after,
            "predictor_calls",
        )
        predictor_s = stat_delta(
            before,
            after,
            "predictor_s",
        )

        requests = hits + misses
        miss_rate = misses / requests if requests else 0.0

        result.update(
            {
                "prompt_id": item["prompt_id"],
                "workload": item["workload"],
                "hits": hits,
                "misses": misses,
                "evictions": evictions,
                "fetch_s": fetch_s,
                "miss_rate": miss_rate,
                "predictor_calls": predictor_calls,
                "predictor_s": predictor_s,
                "predictor_ms_per_call": (
                    predictor_s / predictor_calls * 1000.0
                    if predictor_calls
                    else 0.0
                ),
            }
        )

        prompt_results.append(result)

        print(
            f"[{index:02d}/{len(prompts):02d}] "
            f"{item['prompt_id']:<12} "
            f"{result['wall_tps']:>6.2f} tok/s  "
            f"miss={miss_rate * 100:>5.1f}%  "
            f"fetch={fetch_s:>6.3f}s  "
            f"pred={result['predictor_s']:>6.3f}s"
        )

    final_stats = aggregate_stats(pools)

    total_hits = stat_delta(initial_stats, final_stats, "hits")
    total_misses = stat_delta(initial_stats, final_stats, "misses")
    total_evictions = stat_delta(initial_stats, final_stats, "evictions")
    total_fetch_s = stat_delta(initial_stats, final_stats, "fetch_s")
    total_predictor_calls = stat_delta(
        initial_stats,
        final_stats,
        "predictor_calls",
    )
    total_predictor_s = stat_delta(
        initial_stats,
        final_stats,
        "predictor_s",
    )

    total_requests = total_hits + total_misses
    total_tokens = sum(result["tokens"] for result in prompt_results)
    total_elapsed = sum(result["elapsed_s"] for result in prompt_results)

    aggregate_tps = total_tokens / total_elapsed
    overall_miss_rate = (
        total_misses / total_requests
        if total_requests
        else 0.0
    )

    median_prompt_tps = statistics.median(
        result["wall_tps"]
        for result in prompt_results
    )

    mean_fetch_per_miss_ms = (
        total_fetch_s / total_misses * 1000.0
        if total_misses
        else 0.0
    )

    predictor_ms_per_call = (
        total_predictor_s / total_predictor_calls * 1000.0
        if total_predictor_calls
        else 0.0
    )

    summary = {
        "budget_gib": args.budget_gib,
        "policy": policy,
        "prefetch": prefetch,
        "prompt_count": len(prompts),
        "prompt_ids": [item["prompt_id"] for item in prompts],
        "max_tokens": MAX_TOKENS,
        "total_tokens": total_tokens,
        "total_elapsed_s": total_elapsed,
        "aggregate_tps": aggregate_tps,
        "median_prompt_tps": median_prompt_tps,
        "hits": total_hits,
        "misses": total_misses,
        "evictions": total_evictions,
        "miss_rate": overall_miss_rate,
        "fetch_s": total_fetch_s,
        "mean_fetch_per_miss_ms": mean_fetch_per_miss_ms,
        "predictor_calls": total_predictor_calls,
        "predictor_s": total_predictor_s,
        "predictor_ms_per_call": predictor_ms_per_call,
    }

    output = {
        "summary": summary,
        "prompts": prompt_results,
        "raw_final_stats": final_stats,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))

    print()
    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"Policy:                {policy}")
    print(f"Budget:                {args.budget_gib:.1f} GiB")
    print(f"Aggregate throughput:  {aggregate_tps:.2f} tok/s")
    print(f"Median prompt TPS:     {median_prompt_tps:.2f} tok/s")
    print(f"Hits:                   {total_hits:,}")
    print(f"Misses:                 {total_misses:,}")
    print(f"Miss rate:              {overall_miss_rate * 100:.2f}%")
    print(f"Evictions:              {total_evictions:,}")
    print(f"Fetch time:             {total_fetch_s:.3f} s")
    print(f"Mean fetch/miss:        {mean_fetch_per_miss_ms:.3f} ms")
    print(f"Predictor calls:        {total_predictor_calls:,}")
    print(f"Predictor time:         {total_predictor_s:.3f} s")
    print(f"Predictor ms/call:      {predictor_ms_per_call:.4f} ms")
    print()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
