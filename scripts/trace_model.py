from mlx_lm import load, generate

from moe_trace.tracer import trace_qwen3_moe


MODEL = "mlx-community/Qwen3-30B-A3B-4bit"


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

    print(f"\nCaptured routing events: {len(trace.events)}")

    for event in trace.events[:5]:
        print(
            f"Layer {event.layer}: "
            f"experts={event.expert_ids} "
            f"scores={event.scores}"
        )


if __name__ == "__main__":
    main()