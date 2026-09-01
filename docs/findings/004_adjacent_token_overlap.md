# Finding 004 — Adjacent-Token Expert Overlap

## Question

Do consecutive tokens reuse the same MoE experts?

Does the amount of reuse change between transformer layers?

This question tests whether expert routing has temporal locality.

Temporal locality can make expert caching useful.

If consecutive tokens use unrelated experts, a cache policy based on recent use will have less value.

## Data

The analysis used the first validated routing trace.

The trace contained:

- 33 token positions;
- 48 MoE layers;
- 1,584 token-layer routing events;
- 8 selected experts for each token and layer.

This is a small test.

It contains one prompt.

The result must not be treated as a general measurement of Qwen3 routing.

## Measurement

For two adjacent tokens, expert overlap is:

\[
\text{overlap}(t,t+1)
=
\frac{|E_t \cap E_{t+1}|}{8}
\]

Here, \(E_t\) is the set of 8 experts for token \(t\).

Example:

If two adjacent tokens share 3 experts:

\[
\frac{3}{8}=37.5\%
\]

The analysis calculated this value separately for each layer.

Each layer had 32 adjacent-token comparisons.

## Result

The overall mean adjacent-token overlap was:

**38.9%**

This is equal to approximately:

**3.1 of 8 experts**

that are common to two adjacent routing decisions.

The result was not uniform across layers.

### Layers with high mean overlap

| Layer | Mean overlap |
|---:|---:|
| 6 | 53.5% |
| 7 | 52.7% |
| 30 | 52.3% |
| 20 | 51.2% |
| 33 | 50.4% |
| 18 | 50.0% |
| 31 | 50.0% |

### Layers with low mean overlap

| Layer | Mean overlap |
|---:|---:|
| 0 | 9.8% |
| 47 | 17.6% |
| 2 | 22.3% |
| 37 | 27.3% |
| 41 | 28.1% |
| 42 | 29.3% |

The measured layer means ranged from:

**9.8% to 53.5%**

## Initial interpretation

The first trace contains clear temporal locality.

Consecutive tokens frequently use some of the same experts.

The amount of reuse also changes substantially between layers.

For example:

- Layer 0 had 9.8% mean overlap.
- Layer 6 had 53.5% mean overlap.

Thus, a single cache rule can possibly have different effectiveness at different layers.

A layer with high adjacent-token overlap can benefit more from recent expert retention.

A layer with low adjacent-token overlap can require a different policy.

This result does not prove that a predictive cache will improve inference.

It also does not prove that 38.9% is a typical value for this model.

The trace is too small for these conclusions.

The result only shows that sufficient routing structure exists to justify more tests.

## Important limitation

The current result combines routing events from the first test trace.

Future analysis must separate:

- prompt prefill;
- autoregressive decode.

Decode behavior is especially important for sequential expert caching and prefetch.

The project must also test more prompts and workload types.

## Decision

**CONTINUE**

The decision rule was:

- Stop or reconsider the predictive-cache work if adjacent-token overlap is very low and approximately uniform.
- Continue if overlap is meaningful or strongly layer-dependent.

The first trace had:

- 38.9% overall mean overlap;
- a layer range from 9.8% to 53.5%.

The result meets both continuation conditions.

## Next questions

The next routing analyses are:

1. Are some experts used much more frequently than other experts?
2. How concentrated is expert usage in each layer?
3. How different are prefill and decode routing patterns?
4. Can the current expert set help predict the next expert set?
5. Do these results continue across different prompts and task types?

If these tests show little additional structure, reconsider the cache-optimization work.

If the structure is repeatable, continue to cache simulation.

## Status

Project stage: Routing characterization

Finding: First evidence of temporal expert reuse

Outcome: Continue