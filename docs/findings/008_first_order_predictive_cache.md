# Finding 008 — First-Order Predictive Cache

## Question

Can a simple first-order transition model improve expert cache performance compared with LRU?

The test uses only routing transitions that occurred earlier in the same sequence.

The policy does not use future routing information.

The test measures how much of the LRU-to-oracle gap the transition model can capture.

## Data

The simulation used the cross-workload routing dataset from Finding 006.

The dataset contained:

* 25 prompts;
* 5 workload types;
* 3,225 decode token positions;
* 191,568 routing events;
* 48 MoE layers;
* 128 available experts;
* 8 selected experts for each token and layer.

The simulation used decode events only.

The cache was reset between prompts.

Each MoE layer had an independent cache.

## Baseline policies

The test compared three policies.

### LRU

LRU means Least Recently Used.

The cache keeps experts that were used most recently.

### First-order Markov policy

The Markov policy learns expert transitions from earlier routing events.

For each observed transition, the policy records which experts followed the experts in the current routing set.

The policy then uses these historical transition counts when it decides which experts to keep in the cache.

The policy uses only past observations.

It does not use future routing requests.

### Oracle

The oracle knows the future routing sequence.

It keeps the experts that will be needed soonest.

The oracle is not a practical cache policy.

It is an upper reference for the current simulation model.

## Cache capacities

The test used these cache capacities for each layer:

* 8 experts;
* 16 experts;
* 32 experts.

The 8-expert cache is the tightest tested condition.

It can store the same number of experts as one routing decision selects.

## Results

### Cache capacity: 8 experts

| Policy | Hit rate |    Hits |  Misses |
| ------ | -------: | ------: | ------: |
| LRU    |    42.4% | 524,883 | 713,517 |
| Markov |    47.6% | 588,916 | 649,484 |
| Oracle |    60.8% | 753,417 | 484,983 |

The Markov policy improved the hit rate by:

**5.2 percentage points**

The Markov policy avoided:

**64,033**

misses compared with LRU.

The available LRU-to-oracle gap was:

**18.5 percentage points**

The Markov policy captured:

**28.0%**

of this gap.

### Cache capacity: 16 experts

| Policy | Hit rate |    Hits |  Misses |
| ------ | -------: | ------: | ------: |
| LRU    |    64.7% | 800,857 | 437,543 |
| Markov |    66.2% | 819,263 | 419,137 |
| Oracle |    78.4% | 971,367 | 267,033 |

The Markov policy improved the hit rate by:

**1.5 percentage points**

The Markov policy avoided:

**18,406**

misses compared with LRU.

The available LRU-to-oracle gap was:

**13.8 percentage points**

The Markov policy captured:

**10.8%**

of this gap.

### Cache capacity: 32 experts

| Policy | Hit rate |      Hits |  Misses |
| ------ | -------: | --------: | ------: |
| LRU    |    81.0% | 1,003,095 | 235,305 |
| Markov |    82.0% | 1,015,680 | 222,720 |
| Oracle |    88.8% | 1,099,573 | 138,827 |

The Markov policy improved the hit rate by:

**1.0 percentage point**

The Markov policy avoided:

**12,585**

misses compared with LRU.

The available LRU-to-oracle gap was:

**7.8 percentage points**

The Markov policy captured:

**13.0%**

of this gap.

## Main result

The first-order Markov policy performed better than LRU at all tested cache capacities.

The largest improvement occurred with the smallest cache.

At a capacity of 8 experts per layer:

* LRU achieved 42.4%;
* Markov achieved 47.6%;
* Oracle achieved 60.8%.

The Markov policy captured 28.0% of the available LRU-to-oracle gap.

The improvement was smaller at capacities of 16 and 32 experts.

## Interpretation

The result shows that first-order routing transitions contain useful information.

Simple recency is therefore not the only useful signal.

However, the first-order transition model did not capture most of the available oracle headroom.

At a capacity of 8 experts, 72.0% of the LRU-to-oracle gap remained uncaptured.

At capacities of 16 and 32 experts, the uncaptured share was larger.

This result indicates that pairwise expert transitions are useful but incomplete.

Additional routing history can contain useful information.

## Practical meaning

The Markov policy is a low-cost and interpretable predictor.

It does not require a learned neural model.

It also does not use future routing information.

This makes it more realistic than the oracle.

However, the simulation still measures cache hit rate only.

It does not yet measure real inference speed.

The result therefore does not show a 5.2% or larger runtime improvement.

Real runtime tests must include:

* prediction cost;
* cache-management cost;
* expert transfer cost;
* storage or memory latency;
* asynchronous execution.

## Important limitation

The Markov policy learns transitions inside each prompt and layer.

The current test does not measure whether transition knowledge transfers between prompts.

The test also uses first-order transitions only.

It does not use:

* longer routing history;
* router scores;
* token information;
* workload information;
* hidden-state information.

The policy is therefore a simple baseline.

## Decision

**CONTINUE WITH ONE SHORT-HISTORY TEST**

The decision rule was:

* Stop if the first-order policy does not improve on LRU.
* Continue if the policy produces a repeatable improvement and useful oracle-gap capture.

The Markov policy improved on LRU at all tested capacities.

The largest gain was 5.2 percentage points at a capacity of 8 experts.

This result is sufficient to test one more simple temporal method.

The next method should use short routing history.

Do not add a complex learned model before this test.

## Next question

The next research question is:

> Does a short history of previous routing sets capture substantially more of the oracle gap than a first-order transition model?

The next policy should use a small number of previous routing states.

For example:

* current routing set;
* previous routing set;
* routing set two steps earlier.

If the short-history policy gives only a small improvement over the first-order Markov policy, stop increasing hand-built transition complexity.

At that point, decide whether a learned predictor is justified.

If the short-history policy gives a large improvement, continue with temporal prediction.

## Status

Project stage: Predictive cache simulation

Finding: First-order routing transitions improve cache performance, but most oracle headroom remains

Outcome: Continue with one short-history test

