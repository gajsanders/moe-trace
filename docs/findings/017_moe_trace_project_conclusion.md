# Finding 017 — MoE Trace Project Conclusion

## Question

What did MoE Trace establish?

Did expert-routing structure produce a practical inference optimization?

Should the current experimental program continue?

## Original objective

MoE Trace started with a narrow systems question.

Mixture-of-Experts models activate only a small subset of available experts for each token.

If expert selection has temporal or statistical structure, a runtime can possibly use that structure to improve expert residency, caching, or prefetch.

The project therefore tested the following chain:

1. Can expert routing be observed reliably?
2. Does expert routing contain repeatable structure?
3. Does this structure create theoretical cache headroom?
4. Can causal prediction exploit that headroom?
5. Does improved cache behavior transfer to real MLX execution?
6. Does the resulting optimization improve end-to-end inference speed?

The project used explicit continuation and stop conditions.

The purpose was not to continue developing a predictor indefinitely.

The purpose was to determine whether predictive expert caching was technically and economically useful in the tested runtime.

## Model and runtime

The primary model was:

`mlx-community/Qwen3-30B-A3B-4bit`

The model has:

* 48 MoE layers;
* 128 experts per MoE layer;
* 8 selected experts per token.

The main local runtime was MLX on Apple Silicon.

Constrained-memory runtime validation used streamlx with SSD-backed expert storage.

## Finding 001–003 — Instrumentation and trace validity

The first requirement was reliable routing observation.

MoE Trace successfully instrumented the Qwen3 MoE router without making a permanent modification to the installed MLX-LM source code.

The tracer captured:

* expert identifiers;
* router scores;
* transformer layer;
* token position;
* inference phase.

The normalized validation trace contained:

* 48 layers;
* 33 token positions;
* 1,584 token-layer routing events;
* 8 experts for every token-layer decision.

The instrumentation and normalization stages passed.

The project therefore had a valid basis for routing analysis.

## Finding 004–006 — Routing has substantial structure

The first trace showed meaningful adjacent-token expert reuse.

The overall mean adjacent-token overlap was:

**38.9%**

or approximately:

**3.1 of 8 experts**

shared between consecutive routing decisions.

The amount of overlap varied substantially between layers.

The broader workload suite confirmed that routing structure was not limited to one prompt.

Across multiple workloads and prompts, decode routing showed repeatable temporal locality, expert concentration, and strong layer differences.

This established the first major result:

> MoE expert routing is structured enough to justify cache-policy analysis.

## Finding 007 — LRU leaves theoretical headroom

MoE Trace then compared simple cache policies with an oracle.

LRU did not capture all available routing locality.

At small cache capacities, the oracle had substantial additional hit-rate headroom.

This established that a better cache policy was theoretically possible.

The result justified predictive-cache experiments.

## Finding 008–012 — Prediction works in simulation

A first-order causal Markov policy improved simulated cache performance over LRU.

Longer routing histories added relatively little additional value.

A leakage-safe XGBoost predictor improved the cache result further.

However, feature ablation showed that most of the predictive value came from a very small feature set.

The useful reduced model contained:

* Markov score;
* current router score;
* previous router score.

This model performed approximately as well as the larger 16-feature model.

The project therefore established:

> Future expert use contains causal predictive signal that can improve simulated cache decisions.

This was a positive result.

It did not yet establish runtime value.

## Finding 013 — Expert locality has a real MLX execution cost

The project then tested whether expert access locality changes actual MLX execution latency.

Synthetic expert-access patterns showed a consistent locality effect.

Compared with repeatedly using the same 8 experts:

* alternating across 16 experts was slower;
* rotating through all 128 experts was slower.

Across the tested layers, rotating through all 128 experts produced a median slowdown of approximately:

**8.45%**

This showed that expert locality is not only a statistical property of routing traces.

It has a measurable physical effect in the MLX runtime.

## Finding 014 — Resident-memory locality is real but too small

Real recorded routing was then replayed in natural and shuffled temporal order.

Natural routing was modestly faster.

The median layer-level shuffled penalty was approximately:

**1.46%**

The effect was positive in most tested prompt-layer pairs.

However, the expert path represented only part of total decode time.

The estimated expert-compute share was approximately:

**44.1%**

A 1.46% expert-path locality effect therefore corresponds to only about:

**0.64%**

of whole-token time before any optimization overhead.

Two stronger interventions were also tested.

### Expert order steering

An oracle knew the next expert set and changed the current expert execution order to try to improve future locality.

The median result was:

**+1.68% slower**

and no tested layer improved consistently.

This branch stopped.

### Expert pre-touch

An oracle pre-touched experts before their use.

Warmed expert compute was not faster.

The median warmed-compute result was approximately:

**+5.26% slower.**

The synchronous prefetch path was much worse.

This branch also stopped.

Finding 014 therefore established:

> Resident-memory expert locality exists, but its usable end-to-end value is too small for the tested interventions.

## Finding 015 — Better physical caching can still produce worse inference

The project then moved to a constrained-memory runtime.

streamlx provided:

* SSD-backed expert storage;
* fixed expert-cache capacity;
* real cache hits and misses;
* real fetch latency;
* end-to-end token throughput.

A 4 GiB cache budget produced a meaningful miss rate and therefore created a real constrained-residency problem.

### LRU capacity curve

Increasing cache capacity produced a clear reduction in misses and increase in throughput.

For example:

| Budget |  Throughput | Miss rate |
| -----: | ----------: | --------: |
|  1 GiB |  5.27 tok/s |    58.44% |
|  2 GiB |  7.10 tok/s |    42.69% |
|  4 GiB | 11.00 tok/s |    21.89% |
|  8 GiB | 24.23 tok/s |     7.08% |

This confirmed that expert-cache quality matters materially in the constrained runtime.

### Simple Markov runtime policy

The simple Markov policy did not transfer successfully from simulation.

In the physical cache, it produced worse behavior than LRU.

The full-context Markov version produced approximately:

* 7.34 tok/s;
* 34.49% miss rate.

The LRU reference was approximately:

* 11.00 tok/s;
* 21.89% miss rate.

This branch stopped.

### Router-aware learned policy

The frozen three-feature Markov + Router XGBoost predictor produced a different result.

Compared with the held-out LRU control:

| Metric     |         LRU | Markov + Router |
| ---------- | ----------: | --------------: |
| Throughput | 11.04 tok/s |      9.00 tok/s |
| Miss rate  |      22.21% |          19.86% |
| Misses     |      27,734 |          24,804 |
| Fetch time |    13.901 s |        12.583 s |

The learned policy therefore produced:

* 2,930 fewer physical cache misses;
* approximately 10.6% relative miss-count reduction;
* 1.318 seconds less SSD fetch time.

This showed that the predictive signal transferred into a real physical cache.

However, the predictor required:

* 15,600 XGBoost calls;
* 3.796 seconds of XGBoost inference;
* approximately 0.2433 ms per predictor call.

The saved fetch time was only:

**1.318 seconds.**

The XGBoost inference cost alone was approximately:

**2.9 times**

the saved SSD fetch time.

End-to-end throughput decreased by:

**18.5%.**

This produced the central systems result of MoE Trace:

> A cache policy can make better physical cache decisions and still make the complete inference system slower.

Prediction accuracy and cache-hit improvement are therefore insufficient evaluation targets.

The complete runtime cost of extracting and using the prediction must be included.

## Finding 016 — Near-zero-cost allocation also has insufficient value

Finding 016 tested the opposite strategy.

Instead of using a runtime predictor, it used a static layer-aware memory allocation.

This added effectively no per-token decision cost.

The same total cache capacity was preserved.

The optimizer used training prompts only and selected different capacities for different layers.

On held-out prompts:

* equal allocation produced 44,539 misses;
* layer-aware allocation produced 43,549 misses.

The relative miss reduction was:

**2.22%.**

The pre-defined continuation threshold was:

**5%.**

The result failed the threshold.

Physical runtime validation was not performed.

This completed the final unresolved cache-allocation hypothesis.

## Combined interpretation

The project established four separate facts.

### 1. Routing structure is real

Expert routing has:

* temporal locality;
* expert-frequency concentration;
* substantial layer variation;
* repeatable structure across workloads.

### 2. Predictive signal is real

Causal routing information can predict future expert use.

Simple Markov information is useful.

Router scores add further useful signal.

A small learned model can improve cache decisions.

### 3. Physical expert locality matters

Expert access locality changes real MLX execution cost.

A constrained-memory expert cache also has a strong effect on throughput.

The physical mechanism therefore exists.

### 4. The tested optimization is not economically useful

The problem is the relationship between benefit and control cost.

Methods with enough predictive power to produce a meaningful physical miss reduction introduced too much runtime overhead.

Methods with almost no runtime overhead produced too little improvement.

This can be summarized as:

$$
\text{Net runtime value}
=
\text{avoided execution or I/O cost}
-
\text{prediction cost}
-
\text{synchronization cost}
-
\text{coordination cost}
$$

A successful inference optimization requires this value to remain positive.

MoE Trace did not find a predictive expert-cache mechanism that met this requirement for the tested system.

## Main project result

The main result is not:

**Predictive expert caching does not work.**

That statement would be incorrect.

MoE Trace showed that router-aware prediction can improve a real expert cache.

The correct result is:

> **Routing predictability can improve expert-cache decisions, but improved cache behavior does not guarantee improved end-to-end inference. In the tested MLX and streamlx environments, the runtime cost of using the predictive signal exceeded its recovered execution or I/O value.**

This distinction is important.

## Negative results

The project stopped the following branches:

* resident-memory order steering;
* resident-memory expert pre-touch;
* simple Markov physical eviction;
* runtime XGBoost predictive eviction;
* additional predictor complexity;
* static layer-aware cache allocation.

These negative results are part of the project outcome.

They define where the available optimization headroom disappeared.

## What the project did not establish

MoE Trace does not prove that predictive expert caching is ineffective for all MoE systems.

The result is limited by:

* one primary model family;
* Apple Silicon;
* MLX;
* one constrained-memory implementation;
* one storage and memory hierarchy;
* the tested prompt suite;
* the tested cache budgets.

Different hardware can change the cost balance.

For example, slower expert storage can increase the value of avoiding one miss.

A cheaper integrated predictor can reduce control cost.

Different MoE architectures can have stronger or weaker routing locality.

The project therefore provides a measured systems result, not a universal impossibility result.

## Research value

The project developed and validated a complete experimental chain:

1. instrument routing;
2. validate traces;
3. characterize statistical structure;
4. establish oracle headroom;
5. test causal predictors;
6. remove unnecessary predictor complexity;
7. measure the physical locality mechanism;
8. estimate end-to-end headroom;
9. validate in a constrained-memory runtime;
10. compare proxy improvement with complete-system throughput.

The most important methodological lesson is:

> **Do not stop at prediction accuracy, cache hit rate, or simulated latency. Measure whether the optimization survives its own implementation cost.**

This lesson is broader than MoE expert caching.

## Decision

**STOP the MoE Trace predictive-cache experimental program.**

Do not continue with:

* larger XGBoost models;
* GRU runtime predictors;
* longer routing history;
* workload-specific predictors;
* additional Markov variants;
* additional cache-allocation tuning;
* combined predictive and allocation policies.

The project has answered its original question sufficiently.

Further work on these branches would be optimization rescue work rather than a new test of the original hypothesis.

## Project status

**MoE Trace V1: COMPLETE**

| Area                                 | Result                |
| ------------------------------------ | --------------------- |
| Router instrumentation               | PASS                  |
| Trace validation                     | PASS                  |
| Routing locality                     | ESTABLISHED           |
| Cross-workload robustness            | ESTABLISHED           |
| Cache headroom                       | ESTABLISHED           |
| Causal predictive signal             | ESTABLISHED           |
| Learned predictive signal            | ESTABLISHED           |
| Physical locality effect             | ESTABLISHED           |
| Resident-memory optimization         | STOP                  |
| Simple predictive physical caching   | STOP                  |
| Router-aware physical cache quality  | IMPROVED              |
| Router-aware end-to-end throughput   | FAILED                |
| Static layer allocation              | INSUFFICIENT HEADROOM |
| Further predictive-cache development | STOP                  |

## Final conclusion

MoE Trace began with the hypothesis that expert-routing structure could support better expert caching.

The project confirmed the first half of that hypothesis.

The routing structure exists.

It is measurable.

It is repeatable.

It is predictive.

It can improve a physical cache.

The project rejected the second half for the tested runtime.

The available runtime value was not large enough to pay for the mechanisms required to extract it.

That is the final result.

## Next direction

Any future project should not start by searching for a more accurate expert predictor.

A stronger question is:

**Can trace-derived measurements determine whether an inference optimization has enough end-to-end economic headroom before substantial implementation effort is spent on it?**

MoE Trace can serve as the first case study for that broader systems question.

## Status

Project stage: Final synthesis

Finding: Predictive expert-cache quality can improve without improving inference

Outcome: **MOE TRACE V1 COMPLETE**
