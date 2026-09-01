import json
from dataclasses import asdict
from pathlib import Path

from mlx_lm import generate, load

from moe_trace.tracer import trace_qwen3_moe


MODEL = "mlx-community/Qwen3-30B-A3B-4bit"

PROMPT_PATH = Path("prompts/workload_suite.jsonl")
OUTPUT_PATH = Path("results/workload_routing_trace.jsonl")

MAX_TOKENS = 128


def load_prompts():
    with PROMPT_PATH.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    prompts = load_prompts()

    print(f"Loading model: {MODEL}")
    model, tokenizer = load(MODEL)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_events = 0
    total_decode_tokens = 0

    with OUTPUT_PATH.open("w") as output:
        for index, item in enumerate(prompts, start=1):

            prompt_id = item["prompt_id"]
            workload = item["workload"]

            prompt = "/no_think " + item["prompt"]

            print(
                f"[{index:02}/{len(prompts)}] "
                f"{prompt_id} ({workload})"
            )

            with trace_qwen3_moe(model) as trace:
                generate(
                    model,
                    tokenizer,
                    prompt=prompt,
                    max_tokens=MAX_TOKENS,
                    verbose=False,
                )

            decode_tokens = {
                event.token_index
                for event in trace.events
                if event.phase == "decode"
            }

            prefill_tokens = {
                event.token_index
                for event in trace.events
                if event.phase == "prefill"
            }

            print(
                f"    prefill={len(prefill_tokens)} "
                f"decode={len(decode_tokens)} "
                f"events={len(trace.events)}"
            )

            for event in trace.events:
                record = asdict(event)

                record["prompt_id"] = prompt_id
                record["workload"] = workload
                record["model"] = MODEL

                output.write(json.dumps(record) + "\n")

            total_events += len(trace.events)
            total_decode_tokens += len(decode_tokens)

    print("\nCollection complete")
    print(f"Prompts: {len(prompts)}")
    print(f"Decode tokens: {total_decode_tokens}")
    print(f"Routing events: {total_events}")
    print(f"Output: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
