# Finding 006 — Cross-Workload Routing Robustness

## Question

Do the routing patterns from the first experiments continue across different prompts and workload types?

This test checks whether the earlier routing results were specific to one prompt.

The test also checks whether different workload types produce different routing behavior.

## Data

The robustness dataset contained:

* 25 prompts;
* 5 workload types;
* 3,225 decode token positions;
* 191,568 routing events;
* 48 MoE layers;
* 128 available experts;
* 8 selected experts for each token and layer.

The workload types were:

* coding;
* general knowledge;
* mathematical reasoning;
* planning;
* summarisation.

Each workload contained 5 prompts.

The trace validation found:

* 25 prompts;
* 5 workload types;
* 0 duplicate token-layer records.

The dataset was therefore suitable for cross-prompt analysis.

## Measurements

The analysis calculated these values for each prompt:

* mean decode adjacent-token expert overlap;
* mean normalized routing entropy;
* mean top-16 expert share.

The analysis also calculated:

* mean values for each workload type;
* mean decode overlap for each MoE layer.

The previous single-prompt experiment measured:

`41.5%`

mean decode adjacent-token expert overlap.

The new test checks whether this value remains similar across a larger dataset.

## Per-workload results

The workload-level results were:

| Workload               | Decode overlap | Normalized entropy | Top-16 share |
| ---------------------- | -------------: | -----------------: | -----------: |
| Coding                 |          40.7% |              80.8% |        54.5% |
| General knowledge      |          42.7% |              76.5% |        60.8% |
| Mathematical reasoning |          40.5% |              79.7% |        56.9% |
| Planning               |          39.4% |              76.3% |        62.0% |
| Summarisation          |          50.2% |              61.7% |        89.0% |

The overall mean decode overlap was:

**42.7%**

This value is close to the previous single-prompt result of:

**41.5%**

## Per-prompt overlap

All 25 prompts showed substantial adjacent-token expert overlap.

The lowest-overlap prompt was:

`math_04`

with:

**35.3%**

The highest-overlap prompt was:

`summary_03`

with:

**51.1%**

No prompt produced very low decode overlap.

This result shows that the original overlap observation was not limited to one prompt.

## Layer-level overlap

The larger dataset also reproduced strong differences between layers.

Examples include:

| Layer | Mean decode overlap |
| ----: | ------------------: |
|     0 |                7.9% |
|     6 |               56.2% |
|     7 |               59.7% |
|    18 |               55.6% |
|    19 |               54.0% |
|    30 |               54.0% |
|    31 |               57.7% |
|    47 |               20.6% |

The highest-overlap layer was:

`Layer 7`

with:

**59.7%**

The lowest-overlap layer was:

`Layer 0`

with:

**7.9%**

The layer pattern is consistent with the earlier small trace.

Some middle layers have high adjacent-token reuse.

The first and final layers have lower reuse.

## Workload differences

The routing behavior was not identical across workload types.

Summarisation showed the strongest concentration.

The summarisation workload had:

* 50.2% mean decode overlap;
* 61.7% normalized entropy;
* 89.0% mean top-16 expert share.

The other workload types had lower top-16 shares.

Their values were between:

`54.5%`

and:

`62.0%`

This difference can indicate workload-dependent routing behavior.

However, the current dataset contains only 5 prompts for each workload type.

The result is not sufficient to claim that summarisation always produces more concentrated routing.

More data is required before making that conclusion.

## Interpretation

The main routing patterns survived across prompts and workloads.

Three results reproduced.

First, adjacent-token expert overlap remained substantial.

Second, layer-level routing behavior remained different across the model.

Third, expert concentration remained visible.

The overall decode overlap increased slightly from:

`41.5%`

in the first short trace to:

`42.7%`

in the larger dataset.

This indicates that the first result was not a one-prompt anomaly.

The result also shows that workload type can affect routing concentration.

Summarisation produced a substantially more concentrated expert-use pattern than the other tested workloads.

## Important limitation

The dataset is larger than the first trace, but it is still limited.

It contains:

* 25 prompts;
* 5 prompts for each workload type;
* one MoE model;
* one model checkpoint;
* one hardware and runtime configuration.

The results therefore support a claim about this test configuration.

They do not yet support a general claim about all MoE models.

Future tests can use:

* more prompts;
* more workload types;
* different MoE models;
* different model sizes;
* different runtime conditions.

## Decision

**CONTINUE**

The decision rule was:

* Continue if routing overlap and concentration remain visible across prompts and workloads.
* Reassess if the earlier result disappears in the larger dataset.
* Narrow the claim if only one workload type shows the effect.

The new dataset showed:

* substantial decode overlap in all 25 prompts;
* similar mean overlap across four workload types;
* stronger overlap and concentration in summarisation;
* repeatable layer-level differences.

The result meets the continuation condition.

The next stage can test whether this routing structure produces useful cache-performance differences.

## Next question

The next research question is:

> Does the observed routing structure create a meaningful performance gap between simple cache policies and an oracle cache policy?

The next stage should simulate:

* random cache replacement;
* static expert popularity;
* LFU;
* LRU;
* oracle caching.

If LRU performs close to the oracle, stop the predictive-cache investigation.

If the oracle performs substantially better than LRU, continue to layer-aware and predictive cache policies.

## Status

Project stage: Cross-workload routing validation

Finding: Routing structure survives across prompts and workload types

Outcome: Continue

