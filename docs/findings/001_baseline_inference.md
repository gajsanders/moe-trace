# Finding 001 — Baseline MoE Inference

## Question

Can the selected MoE model run correctly with MLX on Apple Silicon?

A working baseline is necessary before MoE Trace adds instrumentation.

The baseline also gives reference values for memory use and inference speed.

## Test configuration

Model:

`mlx-community/Qwen3-30B-A3B-4bit`

Runtime:

- MLX
- MLX-LM

Python version:

`3.12.14`

MLX device:

`Device(gpu, 0)`

The test ran locally on Apple Silicon.

The model uses a Mixture-of-Experts architecture.

## Test prompt

The test used this prompt:

`/no_think Explain in two sentences why the sky is blue.`

The `/no_think` instruction was used to prevent the model from using most of the generation budget for internal reasoning text.

## Result

The model generated a valid answer.

Measured values were:

| Measurement | Result |
|---|---:|
| Prompt tokens | 21 |
| Generated tokens | 47 |
| Prompt throughput | 166.592 tokens/s |
| Generation throughput | 128.776 tokens/s |
| Peak memory | 17.248 GB |

The model loaded and generated text without an error.

The model used the MLX GPU device.

## Interpretation

The baseline test was successful.

This result shows that the selected model and runtime are suitable for the first MoE Trace experiments.

The measured generation throughput and peak memory provide baseline values.

Later tests can compare instrumented inference with these values.

A large reduction in throughput after instrumentation can indicate that the tracer has too much overhead.

## Decision

**CONTINUE**

The baseline requirement was:

> The model must load and generate a correct response on the target runtime.

The model met this requirement.

The next step was to identify and observe the MoE router.

## Status

Project stage: Baseline inference

Outcome: Pass

Next stage: Router instrumentation
