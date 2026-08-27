# Finding 001 — Adjacent-Token Expert Overlap

## Research question

**Do consecutive tokens in a Mixture-of-Experts model tend to reuse the same experts, and does this behaviour vary by transformer layer?**

This is an important first question for MoE Trace because temporal reuse may create opportunities for expert caching and prefetching. If expert selection were effectively random between consecutive tokens, simple cache policies based on recent usage would have limited value.

## Experimental setup

**Model:** `mlx-community/Qwen3-30B-A3B-4bit`
**Runtime:** MLX / MLX-LM on Apple Silicon
**Experts:** 128
**Experts selected per token:** 8
**MoE layers:** 48

The trace contained:

* 33 token positions
* 48 MoE layers
* 1,584 token-layer routing events
* 576 prefill events
* 1,008 decode events
* 0 missing token-layer pairs
* 0 duplicate token-layer pairs

For every pair of consecutive tokens within each layer, overlap was calculated as:

$$
\text{overlap}(t,t+1)
=
\frac{|E_t \cap E_{t+1}|}{8}
$$

where \(E_t\) is the set of eight experts selected for token \(t\).

## Results

**Overall mean adjacent-token expert overlap: 38.9%**

This corresponds to approximately **3.1 of the 8 selected experts being reused between consecutive tokens on average** in this trace.

There was substantial variation between layers.

### Highest-overlap layers

| Layer | Mean overlap |
| ----: | -----------: |
|     6 |        53.5% |
|     7 |        52.7% |
|    30 |        52.3% |
|    20 |        51.2% |
|    33 |        50.4% |
|    18 |        50.0% |
|    31 |        50.0% |

### Lowest-overlap layers

| Layer | Mean overlap |
| ----: | -----------: |
|     0 |         9.8% |
|    47 |        17.6% |
|     2 |        22.3% |
|    37 |        27.3% |
|    41 |        28.1% |
|    42 |        29.3% |

The full layer-level analysis showed values ranging from **9.8% to 53.5%**.

## Initial interpretation

This first trace shows meaningful temporal locality in MoE expert routing.

Expert selection is also strongly **layer-dependent**. Some layers reuse more than half of their selected experts between consecutive tokens, while others show much weaker persistence.

This suggests that a uniform cache policy applied identically to every layer may not necessarily be optimal.

For example:

* a layer with ~50% adjacent-token reuse may benefit substantially from retaining recently used experts;
* a layer with ~10% reuse may require a different strategy or provide much less benefit from recency-based caching.

However, this result **does not yet establish a general property of Qwen3 routing**.

The experiment contains only one prompt and 33 token positions. The correct conclusion at this stage is:

> **The first observed Qwen3 trace exhibits substantial and strongly layer-dependent adjacent-token expert reuse. This is sufficient evidence to continue investigating routing structure and cache behaviour.**

It would be premature to claim that Qwen3 generally has a 38.9% expert-reuse rate.

## Decision

**CONTINUE.**

The initial decision rule for this experiment was:

* If adjacent-token overlap was negligible and broadly flat across layers, reconsider or stop the predictive-caching investigation.
* If overlap was meaningful or strongly layer-dependent, continue to deeper routing analysis.

The observed result satisfies both continuation criteria:

1. overall overlap was substantial at 38.9%;
2. overlap differed considerably between layers.

## Next questions

The next analyses should determine:

1. **Expert frequency:** Are some experts selected disproportionately often?
2. **Routing concentration / entropy:** Do individual layers rely on relatively small expert working sets?
3. **Prefill vs decode behaviour:** Does the observed temporal locality remain strong during autoregressive decoding?
4. **Transition structure:** Does the current expert set contain information about which experts will be selected next?
5. **Cross-prompt robustness:** Does the pattern persist across different prompts and task categories?

These measurements will determine whether there is enough exploitable structure to justify building the cache simulator and, later, predictive cache policies.

## Status

**MoE Trace stage:** Routing characterisation
**Finding:** 001
**Outcome:** Continue investigation

