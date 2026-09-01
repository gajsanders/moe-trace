# Finding 002 — Router Instrumentation

## Question

Can MoE Trace record the experts that Qwen3 selects during inference without changing the installed MLX-LM source code?

This is a required capability.

Without reliable routing data, the later routing and cache experiments are not possible.

## Router implementation

The Qwen3 MoE implementation is in:

`mlx_lm.models.qwen3_moe`

The relevant class is:

`Qwen3MoeSparseMoeBlock`

The router first calculates gate probabilities.

It then selects the top experts.

The important values are:

- `inds`: selected expert identifiers;
- `scores`: routing scores for the selected experts.

The model has 128 experts.

The router selects 8 experts for each token.

## First instrumentation attempt

The first tracer replaced `__call__` on each MoE block instance.

The model generated a correct response.

However, the tracer recorded:

`0 routing events`

This method did not intercept the MoE calls.

## Cause

Python special-method lookup does not use an instance-level replacement of `__call__` in this case.

The tracer therefore did not receive the calls.

## Second instrumentation method

The second tracer temporarily replaced:

`Qwen3MoeSparseMoeBlock.__call__`

at the class level.

The tracer also created a mapping from each MoE block instance to its transformer layer.

The replacement function reproduced the same forward calculation as the MLX-LM implementation.

It recorded:

- selected expert identifiers;
- routing scores;
- transformer layer.

The tracer restored the original class method after the trace completed.

No change was made to the installed MLX-LM source file.

## Result

The second test recorded:

`1056 routing invocations`

The trace contained valid expert identifiers and routing scores.

Example expert selection:

```text
Layer 0:
[113, 64, 47, 83, 56, 34, 63, 21]

The values were in the expected range for 128 experts.

Each routing decision selected 8 experts.

Interpretation

The instrumentation test was successful.

MoE Trace can observe the internal expert-routing decisions of the selected Qwen3 model.

The instrumentation does not require a permanent change to MLX-LM.

This is important because:

MLX-LM can be upgraded independently;
the MoE Trace code remains separate from the runtime;
the method can be tested and maintained as project code.

The first raw trace stored one event for each MoE block invocation.

This format was sufficient to prove that instrumentation works.

It was not sufficient for later statistical analysis.

The next step was to normalize the data to one record for each token and layer.

Decision

CONTINUE

The instrumentation requirement was:

Record expert selections during real inference without a permanent change to MLX-LM.

The tracer met this requirement.

Status

Project stage: Router instrumentation

## Outcome: Pass

Next stage: Trace normalization and validation