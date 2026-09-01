# Finding 005 — Routing Concentration and Phase Behavior

## Question

Do some experts receive more routing traffic than other experts?

Does expert concentration change between transformer layers?

Does adjacent-token expert overlap remain strong during autoregressive decode?

These questions test whether expert routing has useful statistical structure.

Useful structure can support expert caching and prefetch.

## Data

The analysis used the first validated routing trace.

The trace contained:

* 33 token positions;
* 48 MoE layers;
* 1,584 token-layer routing events;
* 12,672 expert selections;
* 128 available experts;
* 8 selected experts for each token and layer.

This is a small test.

It contains one prompt.

The results must not be treated as general measurements of Qwen3 routing.

## Measurements

The analysis calculated:

* global expert-selection frequency;
* number of unique experts used in each layer;
* normalized routing entropy for each layer;
* selection share of the most-used expert;
* selection share of the top 8 experts;
* selection share of the top 16 experts;
* adjacent-token overlap for prefill;
* adjacent-token overlap for decode.

Normalized entropy uses the maximum possible entropy for 128 experts as the reference.

A higher normalized entropy value indicates more diffuse expert use.

A lower normalized entropy value indicates more concentrated expert use.

## Global expert frequency

All 128 experts were observed.

The most-used global expert was Expert 40.

Expert 40 had:

`266 selections`

This was:

`2.10%`

of all expert selections.

The next most-used experts were:

| Expert | Selections | Share |
| -----: | ---------: | ----: |
|    115 |        235 | 1.85% |
|     96 |        221 | 1.74% |
|     51 |        214 | 1.69% |
|    110 |        171 | 1.35% |

Global expert use was therefore not dominated by one expert.

## Layer routing concentration

The layer-level results showed stronger concentration.

### Most concentrated layer

Layer 18 had:

* 54 unique experts observed;
* 74.0% normalized entropy;
* 7.2% top-1 expert share;
* 41.3% top-8 expert share;
* 66.3% top-16 expert share.

Thus, 16 of the 128 available experts accounted for approximately two-thirds of the observed selections in Layer 18.

### Other concentrated layers

Layer 20 had:

* 59 unique experts observed;
* 74.8% normalized entropy;
* 42.0% top-8 expert share;
* 67.8% top-16 expert share.

Layer 30 had:

* 59 unique experts observed;
* 75.6% normalized entropy;
* 38.6% top-8 expert share;
* 63.3% top-16 expert share.

### Most diffuse layer

Layer 0 had:

* 89 unique experts observed;
* 87.5% normalized entropy;
* 3.8% top-1 expert share;
* 23.9% top-8 expert share;
* 41.3% top-16 expert share.

This shows that expert concentration changed substantially between layers.

## Prefill and decode overlap

Adjacent-token expert overlap was calculated separately for prefill and decode.

The results were:

| Phase   | Mean overlap | Comparisons |
| ------- | -----------: | ----------: |
| Prefill |        36.5% |         528 |
| Decode  |        41.5% |         960 |

Decode overlap was higher than prefill overlap.

During decode, adjacent tokens reused approximately:

`3.3 of 8 experts`

on average.

## Initial interpretation

The first trace shows two forms of routing structure.

First, expert use is concentrated within some layers.

Second, adjacent-token expert reuse remains substantial during decode.

These results are important because decode is the sequential generation phase.

Expert caching and prefetch are most directly relevant during this phase.

The results also show that different layers have different routing behavior.

For example:

* Layer 0 had relatively diffuse expert use.
* Layer 18 had more concentrated expert use.
* Layer 18 also showed substantial adjacent-token overlap in the previous analysis.

This suggests that a cache policy can possibly benefit from layer-specific information.

A global expert-popularity measure can hide this structure.

## Important limitation

The analysis uses one prompt and 33 token positions.

Each layer therefore has only 264 expert selections.

The results are not sufficient to make general claims about the model.

The next test must use more prompts and more decode tokens.

The test should also include different task types.

## Decision

**CONTINUE**

The continuation conditions were:

* expert use is concentrated in at least some layers;
* routing behavior differs between layers;
* decode overlap remains substantial.

The observed trace met all three conditions.

The next stage must test whether these patterns continue across multiple prompts and workloads.

If the patterns do not continue, narrow the conclusion or stop the cache-optimization work.

If the patterns continue, proceed to transition analysis and cache simulation.

## Status

Project stage: Routing characterization

Finding: Layer-specific expert concentration and decode reuse

Outcome: Continue

