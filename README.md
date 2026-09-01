# MoE Trace

**Trace-driven analysis of Mixture-of-Experts routing, cache locality, and runtime optimization on Apple Silicon.**

MoE Trace investigates a simple systems question:

> **Does the routing structure inside a Mixture-of-Experts model contain enough exploitable information to improve expert caching and end-to-end inference performance?**

The project traces real expert selections from Qwen3 on MLX, measures temporal and statistical routing structure, tests cache policies in simulation, validates expert-locality effects on hardware, and finally tests predictive caching in a constrained-memory SSD-backed runtime.

The main result is a useful negative systems result:

> **Router-aware prediction can improve a real expert cache, but improved cache behavior does not necessarily improve inference throughput. In the tested runtime, the cost of extracting and applying the predictive signal exceeded the runtime value it recovered.**

MoE Trace V1 is complete.

---

## Primary model

The main experiments use:

```text
mlx-community/Qwen3-30B-A3B-4bit
```

The model has:

* 48 MoE layers;
* 128 experts per MoE layer;
* top-8 expert routing.

Primary execution environment:

* Apple Silicon;
* MLX / MLX-LM;
* constrained-memory runtime validation with streamlx.

---

## Research path

MoE Trace follows the optimization chain from statistical structure to end-to-end runtime:

```text
Router instrumentation
        ↓
Trace validation
        ↓
Routing locality
        ↓
Cache headroom
        ↓
Causal prediction
        ↓
Learned prediction
        ↓
Feature ablation
        ↓
Physical locality
        ↓
Constrained-memory caching
        ↓
End-to-end throughput
```

The project deliberately uses stop conditions.

A cache policy is not considered useful because it improves prediction accuracy or cache hit rate alone.

It must improve the complete inference system.

---

## Main findings

### 1. Expert routing has strong temporal structure

The first validated trace showed:

* **38.9% mean adjacent-token expert overlap**;
* approximately **3.1 of 8 experts reused** between consecutive routing decisions;
* large differences between transformer layers.

The broader workload suite confirmed that temporal locality and expert concentration persist across prompts and workload types.

See:

* `docs/findings/004_adjacent_token_overlap.md`
* `docs/findings/005_routing_concentration_and_phase.md`
* `docs/findings/006_cross_workload_robustness.md`

---

### 2. LRU leaves measurable cache headroom

Under constrained simulated cache capacity, simple LRU did not capture all available routing locality.

At capacity 8:

```text
LRU hit rate:       42.38%
Oracle hit rate:    60.80%
```

This established that better cache decisions were theoretically possible.

See:

* `docs/findings/007_cache_policy_headroom.md`

---

### 3. Causal routing prediction improves simulated caching

A first-order Markov policy improved over LRU.

Longer routing history added little additional value.

A leakage-safe XGBoost predictor improved further.

Feature ablation then showed that most of the useful signal could be reduced to only three features:

```text
markov_score
current_router_score
previous_router_score
```

The reduced predictor performed approximately as well as the full 16-feature model.

See:

* `docs/findings/008_first_order_predictive_cache.md`
* `docs/findings/010_leakage_safe_xgboost_cache_predictor.md`
* `docs/findings/012_predictor_feature_ablation.md`

---

### 4. Expert locality has a real MLX runtime effect

Synthetic expert-access experiments showed that broader expert working sets were consistently slower.

Across representative layers:

```text
alternate 16 experts vs same 8:   +5.51%
rotate all 128 vs same 8:         +8.45%
```

Real routing replay also showed a smaller but repeatable locality effect.

The median layer-level penalty from shuffling natural routing order was approximately:

```text
+1.46%
```

This demonstrated that expert locality is physically measurable in MLX.

See:

* `docs/findings/013_expert_access_locality_runtime_cost.md`
* `docs/findings/014_resident_memory_locality_end_to_end.md`

---

### 5. Resident-memory optimization had insufficient end-to-end value

The measured expert-locality effect was real, but small relative to complete token-generation time.

The expert path represented approximately:

```text
44.1% of decode time
```

A 1.46% expert-locality effect therefore implied only about:

```text
0.64% whole-token equivalent headroom
```

before adding policy overhead.

Two stronger interventions were tested.

Oracle expert-order steering was slower:

```text
median delta: +1.68%
layers faster: 0 / 5
```

Oracle pre-touch also failed:

```text
median warmed-compute delta: +5.26%
```

The resident-memory optimization branch stopped.

See:

* `docs/findings/014_resident_memory_locality_end_to_end.md`

---

### 6. Predictive caching improved a real physical cache

The strongest runtime test used streamlx as an SSD-backed constrained-memory expert cache.

At a 4 GiB budget, the LRU baseline achieved approximately:

```text
11.04 tok/s
22.21% miss rate
27,734 misses
13.901 s fetch time
```

The frozen three-feature Markov + Router XGBoost policy achieved:

```text
9.00 tok/s
19.86% miss rate
24,804 misses
12.583 s fetch time
```

The predictor therefore produced:

```text
2,930 fewer physical cache misses
~10.6% relative miss reduction
1.318 s less SSD fetch time
```

The cache policy worked.

The inference optimization did not.

The predictor required:

```text
15,600 predictor calls
3.796 s XGBoost inference time
0.2433 ms per call
```

The XGBoost inference cost alone was approximately **2.9× larger than the SSD fetch time it saved**.

End-to-end throughput decreased by:

```text
18.5%
```

This is the central MoE Trace result.

> **A policy can make better physical cache decisions and still make the complete inference system slower.**

See:

* `docs/findings/015_predictive_cache_runtime_validation.md`

The streamlx experimental integration is preserved under:

```text
experiments/streamlx/
```

---

### 7. Static layer-aware allocation also had insufficient headroom

The final experiment tested an almost zero-overhead alternative.

Instead of predicting future experts at runtime, MoE Trace redistributed a fixed total cache budget across layers.

The total capacity remained:

```text
1,584 expert slots
```

Equivalent to:

```text
33 slots × 48 layers
```

The allocation was optimized on 20 training prompts and frozen before evaluation on five held-out prompts.

Held-out result:

```text
Equal allocation misses:       44,539
Layer-aware misses:            43,549
Relative miss reduction:        2.22%
```

The pre-defined continuation threshold was 5%.

The branch therefore stopped before physical runtime implementation.

See:

* `docs/findings/016_layer_aware_cache_allocation.md`

---

## Project conclusion

MoE Trace established four things.

### Routing structure is real

Expert routing contains:

* temporal locality;
* frequency concentration;
* layer-dependent behavior;
* cross-workload regularity.

### Predictive signal is real

Past routing and router scores contain causal information about future expert use.

### Physical cache effects are real

Expert access locality changes MLX execution time, and expert-cache misses materially affect constrained-memory throughput.

### Proxy improvements are not enough

The useful systems quantity is not prediction accuracy or cache hit rate by itself.

It is closer to:

```text
Net runtime value
    =
avoided execution / I/O cost
    -
prediction cost
    -
synchronization cost
    -
coordination cost
```

For the tested system, MoE Trace did not find a predictive expert-cache mechanism with positive end-to-end value.

The strongest predictor improved the physical cache but was too expensive.

The cheapest static optimization was inexpensive but produced too little benefit.

---

## Project status

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
| Resident-memory order steering       | STOP                  |
| Resident-memory pre-touch            | STOP                  |
| Simple Markov physical caching       | STOP                  |
| Router-aware physical cache quality  | IMPROVED              |
| Router-aware end-to-end throughput   | FAILED                |
| Static layer-aware allocation        | INSUFFICIENT HEADROOM |
| Further predictive-cache development | STOP                  |

**MoE Trace V1: COMPLETE**

The complete synthesis is in:

```text
docs/findings/017_project_conclusion.md
```

---

## Repository structure

```text
moe-trace/
├── docs/
│   └── findings/              Research findings and decision records
│
├── experiments/
│   └── streamlx/              Constrained-memory runtime experiment
│
├── prompts/
│   └── workload_suite.jsonl   Multi-workload prompt suite
│
├── scripts/                   Trace collection, analysis and experiments
│
├── src/
│   └── moe_trace/
│       ├── analysis/          Routing and robustness analysis
│       └── cache/             Cache simulation
│
└── results/                   Generated experimental outputs
```

---

## Findings

The research record is intentionally sequential.

Each finding answers one bounded question and records a continuation or stop decision.

Start with:

```text
docs/findings/000_project_scope.md
```

and finish with:

```text
docs/findings/017_project_conclusion.md
```

The final experimental result is:

```text
docs/findings/016_layer_aware_cache_allocation.md
```

Finding 017 is a synthesis document rather than an additional experiment.

---

## Reproducing the routing analysis

Create and activate the project environment, then install the project dependencies.

The main workload suite is:

```text
prompts/workload_suite.jsonl
```

Collect routing traces with:

```bash
python scripts/collect_workload_traces.py
```

Run routing robustness analysis with:

```bash
python scripts/analyse_robustness.py
```

Run cache-policy simulation with:

```bash
python scripts/simulate_cache_policies.py
```

Run the final layer-allocation experiment with:

```bash
python scripts/analyse_layer_allocation.py
```

Some experiments require the Qwen3 model to be available through the local Hugging Face cache.

Runtime experiments also depend on MLX-compatible Apple Silicon hardware.

---

## streamlx runtime experiment

The Stage 15 constrained-memory integration is preserved separately from the upstream runtime.

See:

```text
experiments/streamlx/README.md
```

The directory contains:

* the benchmark harness;
* the streamlx modification patch;
* predictor metadata;
* frozen benchmark results.

The generated XGBoost model is not committed because it is reproducible from the training script.

The predictor can be recreated with:

```bash
python scripts/train_runtime_markov_router.py
```

---

## Scope and limitations

MoE Trace primarily studies:

```text
mlx-community/Qwen3-30B-A3B-4bit
```

on Apple Silicon using MLX.

The conclusions should not be treated as universal results for:

* all MoE architectures;
* CUDA runtimes;
* discrete GPU systems;
* different storage hierarchies;
* different expert sizes;
* different memory budgets.

The project demonstrates the measured cost balance for this system.

Other hardware can change that balance.

---

## Why the negative result matters

MoE systems work often reports intermediate metrics such as:

* expert-prediction accuracy;
* cache hit rate;
* bytes transferred;
* theoretical bandwidth reduction.

Those metrics are useful, but they are not the final objective.

MoE Trace followed the optimization through to physical cache behavior and end-to-end token throughput.

The strongest cache policy improved the metric it was designed to optimize and still made inference slower.

That gap between **proxy improvement** and **system improvement** is the main contribution of the project.

---

## Next direction

MoE Trace itself is complete.

A broader follow-on question is:

> **Can trace-derived measurements determine whether an inference optimization has enough end-to-end runtime headroom before substantial implementation effort is spent on it?**

MoE Trace provides one case study for that broader systems problem.
