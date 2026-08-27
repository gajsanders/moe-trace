import json
from dataclasses import asdict
from pathlib import Path

from mlx_lm import load, generate

from moe_trace.tracer import trace_qwen3_moe


MODEL = "mlx-community/Qwen3-30B-A3B-4bit"

OUTPUT_PATH = Path("results/routing_trace.jsonl")


def main():
    model, tokenizer = load(MODEL)

    with trace_qwen3_moe(model) as trace:
        response = generate(
            model,
            tokenizer,
            prompt="/no_think Explain in two sentences why the sky is blue.",
            max_tokens=20,
            verbose=False,
        )

    print("\nMODEL OUTPUT:")
    print(response)

    print(f"\nCaptured token-layer routing events: {len(trace.events)}")

    print("\nFirst 10 routing events:")

    for event in trace.events[:10]:
        print(
            f"phase={event.phase:<7} "
            f"token={event.token_index:<3} "
            f"layer={event.layer:<2} "
            f"experts={event.expert_ids}"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w") as f:
        for event in trace.events:
            f.write(json.dumps(asdict(event)) + "\n")

    print(f"\nSaved routing trace to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()