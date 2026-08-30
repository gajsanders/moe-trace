# Finding 014 — Resident-Memory Expert Locality Has Limited End-to-End Upside

## Question

Can naturally occurring MoE routing patterns be exploited to improve end-to-end inference without changing which experts the model selects?

The objective is to determine whether the routing-locality signal found in earlier experiments can produce a useful runtime improvement in stock MLX.

The investigation had four parts:

* replay real routing sequences;
* test expert-order steering;
* test oracle expert prefetch;
* estimate the expert-path share of end-to-end decode time.

## Model

The investigation used:

`mlx-community/Qwen3-30B-A3B-4bit`

The model contains:

* 48 MoE layers;
* 128 experts per MoE layer;
* top-8 expert routing.

All experiments used the normal MLX resident-memory execution path.

No explicit expert offloading or SSD-backed cache was used.

## 14A — Real routing locality replay

The first test asked whether the natural temporal order of real Qwen3 routing decisions has runtime value.

The benchmark used routing traces from 25 prompts across five workloads.

The selected layers were:

* layer 0;
* layer 7;
* layer 18;
* layer 31;
* layer 47.

For each prompt and layer, the benchmark compared:

* the natural recorded expert-selection order;
* the same expert selections in shuffled temporal order.

Both conditions therefore used:

* the same expert sets;
* the same expert frequencies;
* the same number of calls;
* the same amount of computation.

Only the temporal order changed.

## 14A results

Across 125 prompt-layer pairs:

* shuffled routing was slower on **96 of 125** pairs;
* median paired shuffled penalty was **+0.97%**;
* median layer-level shuffled penalty was **+1.46%**.

Layer-level results were:

| Layer | Natural vs shuffled |
| ----: | ------------------: |
|     0 |              -0.23% |
|     7 |              +1.78% |
|    18 |              +1.46% |
|    31 |              +1.97% |
|    47 |              +0.28% |

A positive value means shuffled routing was slower.

The strongest effect appeared in layers 7, 18, and 31.

These layers also showed stronger routing locality in earlier findings.

## 14A interpretation

Real Qwen3 routing order contains some runtime-useful temporal locality.

The effect is smaller than the synthetic locality effect measured in Finding 013.

Finding 013 compared deliberately extreme access patterns.

Finding 014A used natural model routing.

The smaller effect is therefore expected.

The result supports continued investigation, but it does not show an end-to-end speed improvement.

## 14B — Expert-order steering

The second test asked whether the same eight selected experts could be reordered to improve runtime without changing model semantics.

The model still used:

* the same experts;
* the same router scores;
* the same final expert outputs.

Only the execution order of the selected experts changed.

Three conditions were tested:

* natural expert order;
* numerically sorted expert order;
* oracle next-token-aware order.

The oracle condition knew which current experts would also be used by the next token.

Experts that would be reused were moved to the end of the current execution order.

The expert outputs were then restored to the original order before the normal score-weighting step.

A numerical equivalence test produced:

**maximum absolute difference = 0.0**

This confirmed that the reordering mechanism preserved the tested output.

## 14B results

The oracle policy did not improve performance.

Layer-level oracle deltas were:

| Layer | Oracle order vs natural |
| ----: | ----------------------: |
|     0 |                  +0.61% |
|     7 |                  +2.24% |
|    18 |                  +1.98% |
|    31 |                  +1.68% |
|    47 |                  +1.66% |

The median oracle delta was:

**+1.68%**

The oracle ordering was faster on:

**0 of 5 layers**

The sorted condition was also slower than natural ordering.

## 14B interpretation

Expert-order steering does not provide a useful runtime lever in the tested MLX path.

Even perfect future knowledge did not improve performance.

The additional reorder and restore operations were sufficient to remove any possible locality benefit.

## 14B decision

**STOP expert-order steering**

Do not build a learned predictor for this intervention.

If an oracle cannot produce a gain, a deployable predictor is not justified.

## 14C — Oracle expert prefetch

The third test asked whether selected expert weights could be touched in advance to reduce the latency of the following `SwitchGLU` call.

The oracle prefetch knew the exact experts required by the following computation.

The benchmark touched the selected expert parameter slices for:

* gate projection;
* up projection;
* down projection.

The benchmark measured:

* baseline `SwitchGLU` latency;
* prefetch cost;
* subsequent warmed `SwitchGLU` latency;
* total prefetch plus compute latency.

## 14C results

The prefetch did not reduce subsequent expert computation latency.

Layer-level warmed-compute deltas were:

| Layer | Warmed compute vs baseline |
| ----: | -------------------------: |
|     0 |                     +0.28% |
|     7 |                     +5.81% |
|    18 |                     +5.26% |
|    31 |                     +5.29% |
|    47 |                     +2.35% |

The median warmed-compute delta was:

**+5.26%**

Warmed compute was faster on:

**0 of 5 layers**

The synchronous prefetch itself cost approximately:

**0.52–0.54 ms**

The normal expert computation cost approximately:

**0.17–0.18 ms**

The median total prefetch-path delta was:

**+309.38%**

## 14C interpretation

Explicit pre-touching did not make the later expert operation cheaper.

Instead, the later computation was usually slower.

The experiment does not identify the exact hardware cause.

Possible causes include:

* additional memory-bandwidth use;
* disturbed processor cache state;
* extra MLX scheduling work;
* unfavorable interaction with quantized gather execution.

The important result is operational.

There was no warmed-compute benefit to overlap asynchronously.

## 14C decision

**STOP resident-memory prefetch**

Do not continue to asynchronous prefetch for this model and runtime.

Asynchronous overlap is useful only when the prefetch creates a later benefit that can compensate for its cost.

This condition was not observed.

## 14D — End-to-end expert-path share

The fourth test estimated how much of total decode time is attributable to MoE expert computation.

The benchmark compared:

* normal Qwen3 generation;
* a timing-only bypass condition.

The bypass condition retained the router calculation but removed the expensive `SwitchGLU` expert computation.

The bypass output was not semantically valid.

The condition was used only to estimate runtime cost.

The benchmark used:

* 128 generated tokens;
* 2 warm-up runs;
* 8 measured runs.

## 14D results

Median normal decode time was:

**8.790 ms/token**

Median bypass decode time was:

**4.912 ms/token**

Estimated expert-path cost was:

**3.878 ms/token**

Estimated expert-path share was:

**44.1%**

The expert path therefore represents a substantial part of decode time.

## Locality upside estimate

Finding 14A measured a median layer-level natural-routing locality effect of:

**1.46%**

Finding 14D estimated that expert computation represents:

**44.1%**

of whole-token decode time.

A simple upper-level multiplication gives:

`44.1% × 1.46% ≈ 0.64%`

The approximate whole-token equivalent is therefore:

**0.64%**

This value is not a measured end-to-end speed improvement.

It is an approximate scale estimate.

It does not include the cost of a runtime policy.

## Main result

Resident-memory MoE expert locality is real but difficult to exploit profitably in stock MLX.

The investigation established that:

* natural Qwen3 routing contains measurable temporal locality;
* destroying that locality produces a small runtime penalty;
* changing the execution order of the same experts does not improve performance;
* explicit oracle prefetch does not improve performance;
* expert computation accounts for a substantial part of decode time;
* the realistic locality opportunity in the tested resident-memory path appears to be below 1% of whole-token latency before policy overhead.

## Interpretation

The result does not mean that MoE expert locality is unimportant.

Finding 013 showed a strong runtime difference between deliberately narrow and broad expert working sets.

Finding 014A also showed that real routing order contains useful locality.

The limitation is the runtime environment.

The tested model fits in unified memory.

No expert has to be loaded from SSD or another explicit slow storage tier.

The remaining locality cost is therefore relatively small.

The two tested semantics-preserving interventions introduced more overhead than the locality benefit they attempted to capture.

## Relationship to earlier findings

The project now has the following evidence chain:

```text
routing has temporal structure
        ↓
future expert use is partly predictable
        ↓
cache policies improve simulated hit rates
        ↓
expert working-set locality affects MLX runtime
        ↓
real routing contains a smaller locality benefit
        ↓
resident-memory interventions fail to produce a net gain
```

The predictive-cache work therefore remains technically valid.

However, its strongest application is unlikely to be a model that already fits comfortably in unified memory.

## Practical meaning

The current runtime branch should not be developed further for this resident-memory Qwen3 configuration.

Further work on:

* more complex predictors;
* expert-order steering;
* resident-memory prefetch;
* additional micro-optimizations

is unlikely to provide enough end-to-end benefit to justify the complexity.

The more promising use case is a runtime with a real expert-residency constraint.

Examples can include:

* an MoE model larger than available fast memory;
* an SSD-backed expert store;
* explicit expert offloading;
* a constrained-memory runtime;
* a multi-tier expert cache.

In those environments, an avoided expert miss can have a much larger cost.

The cache-policy results from Findings 007 to 012 can then become more valuable.

## Important limitation

The 14D bypass benchmark is an A/B cost estimate.

It is not exact kernel profiling.

Bypassing expert computation changes hidden states.

This can also change later routing behavior.

The estimated 44.1% expert-path share should therefore be treated as an approximate runtime decomposition.

The 0.64% whole-token equivalent is also an estimate.

It must not be presented as a measured speedup.

## Decision

**STOP the stock resident-memory optimization branch**

Do not continue to more complex locality interventions on this model.

Preserve the routing, cache-simulation, and predictor work.

Revisit these policies when there is a genuine expert-residency or offloading boundary.

## Next question

The next research question is:

> Does predictive expert caching provide meaningful runtime benefits when expert residency is genuinely constrained?

A future experiment should use a runtime where:

* not all expert weights remain resident;
* a cache miss has a measurable load cost;
* expert residency can be explicitly controlled;
* LRU and predictive policies can be compared using real wall-clock inference.

The most useful comparison would be:

```text
LRU expert cache
        vs
Markov expert cache
        vs
Markov + router-score policy
```

using the exact same routing decisions.

The primary metric should be:

**end-to-end tokens per second**

Secondary metrics should include:

* expert cache hit rate;
* expert load count;
* bytes transferred;
* miss latency;
* policy overhead;
* memory use.

## Status

Project stage: Resident-memory runtime investigation complete

Finding: Real routing locality exists, but the tested stock-MLX interventions do not convert it into a useful end-to-end speedup

Estimated expert-path share: **44.1%**

Approximate locality-related whole-token opportunity: **0.64% before policy overhead**

Decision: **STOP resident-memory optimization**

Next direction: constrained-memory or offloaded MoE runtime

