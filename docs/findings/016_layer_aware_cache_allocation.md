# Finding 016 — Layer-Aware Cache Allocation

## Question

Can MoE Trace improve expert-cache efficiency without a runtime prediction model?

Earlier findings showed that routing behavior changes substantially between transformer layers.

Some layers have high expert reuse.

Other layers have low expert reuse.

The Stage 15 runtime experiment also showed that a predictive cache policy can reduce physical cache misses but still reduce end-to-end throughput because the prediction cost is too high.

This finding tests a lower-cost alternative.

The question is:

**Can a fixed total cache capacity perform better if MoE Trace allocates different cache capacities to different transformer layers?**

The allocation is static.

It adds no per-token prediction.

It adds no router-score processing during inference.

It adds no additional synchronization during decode.

## Hypothesis

The earlier routing analysis found large differences between layers.

For example, adjacent-token expert overlap varied substantially across the model.

This suggests that equal cache capacity for every layer can be inefficient.

Some layers can require more capacity.

Other layers can require less capacity.

If this is true, MoE Trace can redistribute a fixed total expert-cache budget across layers and reduce cache misses without adding runtime policy cost.

## Experimental design

Stage 16A used the existing runtime-faithful LRU simulator.

The simulator protects all experts required by the current routing step.

This behavior is closer to the physical streamlx cache than the original abstract LRU simulator.

The experiment used:

* 48 MoE layers;
* 128 experts per layer;
* top-8 expert routing;
* 20 training prompts;
* 5 held-out prompts;
* decode routing only.

The held-out prompts were:

* `coding_05`;
* `general_05`;
* `math_05`;
* `planning_05`;
* `summary_05`.

These are the same held-out prompts used for the Stage 15 runtime gate.

## Fixed total capacity

The Stage 15 4 GiB streamlx baseline used approximately:

**33 expert slots per layer**

for 48 MoE layers.

Stage 16A therefore used:

$$
33 \times 48 = 1584
$$

total expert slots.

The equal-allocation baseline assigned:

**33 slots to every layer.**

The layer-aware condition used exactly the same total:

**1,584 slots.**

The minimum allowed capacity for one layer was:

**8 experts**

because the router selects 8 experts for each token.

The maximum allowed capacity was:

**128 experts.**

## Allocation method

For every layer, the experiment measured the training-set runtime-faithful LRU miss count for capacities from:

**8 to 128 experts.**

It then used exact dynamic programming to find the layer-capacity allocation with the lowest total training-set miss count.

The optimization had one hard constraint:

**The sum of all layer capacities must remain 1,584.**

The held-out prompts were not used to select the allocation.

After the allocation was selected, it was frozen.

The experiment then evaluated the frozen allocation on the five held-out prompts.

## Pre-defined decision rule

The continuation rule was defined before the result was measured.

If the relative held-out miss reduction was:

* **10% or more:** strong pass;
* **5% to 10%:** pass;
* **less than 5%:** stop.

A pass would justify Stage 16B physical runtime validation in streamlx.

A result below 5% would stop the branch before runtime implementation.

## Optimised allocation

The selected capacities were:

| Layer | Capacity | Change from 33 |
| ----: | -------: | -------------: |
|     0 |       49 |            +16 |
|     1 |       36 |             +3 |
|     2 |       38 |             +5 |
|     3 |       36 |             +3 |
|     4 |       35 |             +2 |
|     5 |       34 |             +1 |
|     6 |       27 |             -6 |
|     7 |       25 |             -8 |
|     8 |       29 |             -4 |
|     9 |       29 |             -4 |
|    10 |       32 |             -1 |
|    11 |       34 |             +1 |
|    12 |       31 |             -2 |
|    13 |       40 |             +7 |
|    14 |       31 |             -2 |
|    15 |       28 |             -5 |
|    16 |       34 |             +1 |
|    17 |       36 |             +3 |
|    18 |       26 |             -7 |
|    19 |       27 |             -6 |
|    20 |       30 |             -3 |
|    21 |       31 |             -2 |
|    22 |       32 |             -1 |
|    23 |       37 |             +4 |
|    24 |       30 |             -3 |
|    25 |       40 |             +7 |
|    26 |       35 |             +2 |
|    27 |       30 |             -3 |
|    28 |       32 |             -1 |
|    29 |       37 |             +4 |
|    30 |       26 |             -7 |
|    31 |       26 |             -7 |
|    32 |       31 |             -2 |
|    33 |       30 |             -3 |
|    34 |       32 |             -1 |
|    35 |       33 |              0 |
|    36 |       34 |             +1 |
|    37 |       41 |             +8 |
|    38 |       35 |             +2 |
|    39 |       32 |             -1 |
|    40 |       33 |              0 |
|    41 |       38 |             +5 |
|    42 |       37 |             +4 |
|    43 |       29 |             -4 |
|    44 |       31 |             -2 |
|    45 |       31 |             -2 |
|    46 |       35 |             +2 |
|    47 |       39 |             +6 |

The allocation range was:

**25 to 49 experts per layer.**

The total remained:

**1,584 expert slots.**

The optimizer therefore made substantial changes to the equal allocation.

It did not simply reproduce 33 slots for every layer.

## Training result

| Metric    | Equal allocation | Layer-aware allocation |              Difference |
| --------- | ---------------: | ---------------------: | ----------------------: |
| Hit rate  |           81.59% |                 81.92% | +0.34 percentage points |
| Miss rate |           18.41% |                 18.08% | -0.34 percentage points |
| Misses    |          182,427 |                179,097 |                  -3,330 |

The relative training-set miss reduction was:

**1.83%**

## Held-out result

| Metric    | Equal allocation | Layer-aware allocation |              Difference |
| --------- | ---------------: | ---------------------: | ----------------------: |
| Hit rate  |           82.02% |                 82.42% | +0.40 percentage points |
| Miss rate |           17.98% |                 17.58% | -0.40 percentage points |
| Misses    |           44,539 |                 43,549 |                    -990 |

The relative held-out miss reduction was:

**2.22%**

## Interpretation

Static layer-aware allocation improved cache efficiency.

The improvement also transferred to held-out prompts.

However, the size of the improvement was small.

The optimizer moved substantial capacity between layers.

For example:

* layer 0 increased from 33 to 49 slots;
* layer 7 decreased from 33 to 25 slots;
* layer 13 increased from 33 to 40 slots;
* layer 37 increased from 33 to 41 slots;
* layer 47 increased from 33 to 39 slots.

Despite these large allocation changes, the held-out miss rate improved by only:

**0.40 percentage points.**

The relative held-out miss reduction was only:

**2.22%.**

This result suggests that layer-level routing differences are real but do not create large memory-allocation headroom at this cache budget.

The equal per-layer allocation is therefore already close to a useful operating point for this workload and capacity regime.

## Relationship to earlier findings

Earlier findings showed strong layer differences in expert reuse and concentration.

Those results justified this experiment.

However, layer heterogeneity alone does not imply that unequal cache allocation will produce a large cache benefit.

Finding 015 showed the opposite trade-off.

The router-aware XGBoost policy produced a meaningful physical miss reduction, but its runtime cost was too high.

Finding 016 tested the other extreme:

* almost no runtime policy cost;
* static allocation;
* plain LRU eviction.

The result was also insufficient.

The optimization was cheap, but the available benefit was too small.

Together, Findings 015 and 016 show an important constraint.

A useful MoE cache optimization must provide both:

1. enough improvement in expert residency or fetch behavior; and
2. sufficiently low implementation and decision cost.

A method that satisfies only one condition is not sufficient.

## Limits

This experiment tested one model:

`mlx-community/Qwen3-30B-A3B-4bit`

It tested one routing-trace suite.

It tested one total cache-capacity regime that corresponds approximately to the Stage 15 4 GiB streamlx configuration.

Different models, expert sizes, memory budgets, storage devices, or routing distributions can produce different results.

The result does not prove that layer-aware allocation is never useful.

It shows that the measured opportunity was too small for this project and this experimental configuration.

## Decision

**STOP**

The pre-defined continuation threshold was:

**at least 5% relative held-out miss reduction.**

The measured result was:

**2.22%.**

The result did not meet the threshold.

Stage 16B physical runtime validation will not be performed.

The project will not tune:

* the minimum layer capacity;
* the total cache capacity;
* the allocation optimizer;
* workload-specific allocations;
* per-prompt allocations;
* joint allocation and Markov policies;
* joint allocation and XGBoost policies.

These changes would be post-result attempts to rescue the hypothesis.

## Main result

Static layer-aware expert-cache allocation reduced held-out runtime-faithful LRU misses by:

**2.22%**

under the same total cache capacity.

The improvement was real but too small to justify physical runtime implementation.

## Status

Project stage: Final cache-allocation experiment

Finding: Layer-aware static allocation has insufficient headroom

Outcome: **STOP**

Next stage: Project synthesis and closure
