# Finding 015 — Predictive Cache Gains Do Not Yet Translate to Higher Runtime Throughput

## Question

Can MoE Trace use observed routing structure to improve real end-to-end inference when expert residency is constrained?

Stage 15 tested this question in an SSD-backed MoE runtime.

The experiment used `streamlx` with:

* `mlx-community/Qwen3-30B-A3B-4bit`
* 128 experts
* top-k = 8
* 48 MoE layers
* approximately 15.19 GiB of expert parameters
* streamlx expert streaming from SSD
* explicit expert-memory budgets
* streamlx predictive prefetch disabled

The primary runtime metric was end-to-end generation throughput.

Secondary metrics were:

* expert-cache miss rate
* cache misses
* evictions
* expert fetch time
* predictor evaluation time

## 15A — Constrained-Memory Runtime Compatibility

The existing Qwen3 checkpoint was loaded through streamlx with a 4 GiB expert-memory budget.

Generation completed successfully and produced coherent text.

The runtime reported:

* hits: 19,270
* misses: 5,690
* evictions: 5,690
* miss rate: 22.80%
* expert fetch time: 3.859 s
* predictive prefetch calls: 0

The mean reported fetch time was approximately 0.68 ms per miss in this smoke test.

### Interpretation

This established a real constrained-memory test environment.

A cache miss now caused an actual expert fetch instead of only a simulated penalty.

**Decision: PASS.**

Continue to controlled runtime benchmarking.

---

## 15B — LRU Capacity Baseline

The next experiment measured streamlx LRU behavior at four expert-memory budgets.

The benchmark used:

* 10 fixed prompts
* 64 generated tokens per prompt
* prefetch disabled
* a fresh Python process for each memory budget

### Results

| Expert budget |  Throughput | Miss rate |  Misses | Fetch time | Mean fetch/miss |
| ------------- | ----------: | --------: | ------: | ---------: | --------------: |
| 1 GiB         |  5.27 tok/s |    58.44% | 145,873 |   73.497 s |        0.504 ms |
| 2 GiB         |  7.10 tok/s |    42.69% | 106,549 |   50.380 s |        0.473 ms |
| 4 GiB         | 11.00 tok/s |    21.89% |  54,636 |   27.381 s |        0.501 ms |
| 8 GiB         | 24.23 tok/s |     7.08% |  19,146 |    9.399 s |        0.491 ms |

### Main result

Cache capacity had a large effect on end-to-end inference throughput.

As the expert-memory budget increased:

* cache misses decreased;
* SSD fetch time decreased;
* generation throughput increased.

Mean fetch cost remained close to 0.5 ms per miss across the tested budgets.

### Interpretation

The constrained-memory runtime exposes substantial cache-policy headroom.

A policy that reduces physical cache misses could plausibly improve end-to-end inference.

**Decision: PASS.**

Continue to predictive-cache policies.

---

## 15C — First-Order Markov Eviction

Finding 008 showed that a simple causal Markov cache policy improved simulated cache hit rate over LRU.

Stage 15C attempted to transfer that policy into streamlx.

Several implementation mismatches were identified during the port.

The runtime version was corrected to:

* learn only previously observed routing transitions;
* reset predictor history between prompts;
* include historical expert frequency;
* protect experts required by the current MoE operation;
* observe the complete current top-k routing set before streamlx split the request into cache hits and misses.

A runtime-faithful simulator was also added.

### Runtime-faithful simulation

The revised simulator protected the current routing set from eviction.

Results were:

| Capacity | Original Markov gain over LRU | Runtime-faithful Markov gain |
| -------: | ----------------------------: | ---------------------------: |
|        8 |                      +5.17 pp |                     +0.00 pp |
|       16 |                      +1.49 pp |                     +2.85 pp |
|       32 |                      +1.02 pp |                     +1.43 pp |

Protecting the current routed experts therefore did not explain the expected Markov advantage at capacities 16 and 32.

### Full-context runtime result

At a 4 GiB expert-memory budget:

| Metric     |         LRU |     Markov |
| ---------- | ----------: | ---------: |
| Throughput | 11.00 tok/s | 7.34 tok/s |
| Miss rate  |      21.89% |     34.49% |
| Misses     |      54,636 |     86,076 |
| Fetch time |    27.381 s |   46.087 s |

Relative to LRU:

* throughput decreased by approximately 33%;
* miss rate increased by 12.60 percentage points;
* misses increased by 31,440;
* fetch time increased by 18.706 s.

### Interpretation

The first-order Markov policy did not transfer from simulation to the physical streamlx cache.

The difference was not caused only by predictor computation overhead.

The Markov policy itself produced substantially worse cache decisions.

This also showed that the abstract cache simulator does not fully reproduce the physical cache dynamics of streamlx.

Possible differences include:

* prefill-warmed cache state;
* per-miss physical slot assignment;
* hit/miss overlap behavior;
* coalesced expert reads;
* persistent runtime cache state;
* differences between post-request cache retention and physical working storage.

These mechanisms were not isolated individually in this experiment.

### Decision

**STOP the first-order Markov branch.**

Do not add further hand-built Markov variants.

---

## 15D — Router-Aware Learned Eviction

Findings 010–012 showed that the useful learned signal could be reduced to three features:

* Markov score
* current router score
* previous router score

A frozen XGBoost model was trained using the same configuration as Finding 012.

The training set contained:

* 20 prompts
* 1,966,080 training rows

Five prompts were held out:

* `coding_05`
* `general_05`
* `math_05`
* `planning_05`
* `summary_05`

The model predicted whether an expert would be selected on the next decode token.

The runtime integration used the actual normalized Qwen router scores.

No router scores were approximated or recomputed.

The model was used only for cache eviction decisions.

streamlx predictive prefetch remained disabled.

### Held-out LRU control

At 4 GiB:

* throughput: 11.04 tok/s
* miss rate: 22.21%
* misses: 27,734
* fetch time: 13.901 s

### Markov + Router result

At the same 4 GiB budget:

* throughput: 9.00 tok/s
* miss rate: 19.86%
* misses: 24,804
* fetch time: 12.583 s
* predictor calls: 15,600
* predictor time: 3.796 s
* mean predictor time: 0.2433 ms per call

### Comparison

| Metric     |         LRU | Markov + Router |   Change |
| ---------- | ----------: | --------------: | -------: |
| Throughput | 11.04 tok/s |      9.00 tok/s |  -18.50% |
| Miss rate  |      22.21% |          19.86% | -2.35 pp |
| Misses     |      27,734 |          24,804 |   -2,930 |
| Fetch time |    13.901 s |        12.583 s | -1.318 s |

The learned policy reduced physical cache misses by approximately 10.6%.

This is a real cache-quality improvement.

However, the predictor required 3.796 s of measured XGBoost evaluation time while reducing reported expert fetch time by only 1.318 s.

The predictor therefore cost approximately 2.9 times more than the measured I/O time it saved.

The measured predictor time also does not include all policy overhead.

Additional costs include:

* feature construction;
* Python data structures;
* router-score materialization;
* candidate ranking;
* additional synchronization.

### Break-even estimate

The experiment saved approximately 1.318 s across 15,600 predictor calls.

The predictor would therefore need to cost less than approximately:

0.0845 ms per call

to match the measured fetch-time saving.

The measured XGBoost cost was:

0.2433 ms per call.

This estimate is only a break-even estimate.

A useful optimization would require additional margin after all policy overhead.

---

## Main Result

Predictive expert caching produced two different outcomes.

First-order Markov prediction did not transfer successfully into the physical cache.

Router-aware learned prediction did transfer in terms of cache quality.

The Markov + router model:

* reduced real cache misses;
* reduced real SSD fetch time;
* but reduced end-to-end throughput because policy evaluation was too expensive.

The important distinction is:

> Routing predictability can improve physical cache decisions without improving end-to-end inference.

A cache policy must therefore be evaluated against its complete runtime cost.

Simulated hit-rate improvement alone is not sufficient evidence of a useful MoE runtime optimization.

---

## Relationship to Earlier Findings

Findings 007–012 showed:

* substantial simulated cache-policy headroom;
* useful first-order routing transitions;
* modest gains from learned prediction;
* little value from additional history or larger feature sets;
* Markov score and router scores were sufficient for the best minimal learned model.

Finding 015 qualifies those results.

The simulated Markov advantage did not survive the real runtime.

The router-aware learned signal did survive, but its implementation cost exceeded its physical I/O saving.

Finding 014 had already shown that resident-memory locality offered limited end-to-end upside.

Finding 015 extends that result into the constrained-memory regime:

> Expensive cache misses create real optimization headroom, but prediction overhead can consume that headroom before it becomes throughput.

---

## Practical Meaning

MoE Trace should not continue by adding more predictive model complexity.

Do not proceed to:

* GRU predictors;
* larger XGBoost models;
* additional history features;
* workload-specific predictors;
* repeated hand-built cache heuristics.

Those branches are not justified by the measured runtime results.

One narrower future hypothesis remains credible:

> A substantially cheaper router-aware eviction rule may retain some of the observed cache-quality gain.

This is not part of the current branch.

It should only be revisited as a separate research question if policy evaluation can be made materially cheaper than the current XGBoost implementation.

---

## Important Limitations

This finding applies to:

* Qwen3-30B-A3B-4bit;
* MLX on Apple Silicon;
* the tested streamlx runtime;
* SSD-backed expert streaming;
* the tested expert-memory budgets and prompt suite.

The result does not prove that predictive expert caching cannot work on other:

* MoE architectures;
* storage systems;
* accelerators;
* cache organizations;
* runtimes.

The experiment also did not isolate every cause of the difference between simulated and physical Markov behavior.

---

## Decision

**STOP the current predictive expert-caching optimization branch.**

Preserve:

* routing instrumentation;
* cache simulation;
* runtime-faithful simulation;
* constrained-memory benchmarks;
* the frozen minimal predictor;
* runtime results.

Do not increase predictor complexity.

The router-aware result should remain available as evidence that useful cache signal exists, but the current method does not convert that signal into higher end-to-end throughput.

## Next Question

Review MoE Trace at the project level.

Ask:

> Is there another independent MoE systems hypothesis with enough evidence to justify a new experiment, or has MoE Trace reached a useful natural stopping point?

Any next branch should have its own explicit hypothesis and stop condition.

## Status

**Predictive caching: STOP.**

**MoE Trace project: REVIEW.**
