# Finding 009 — Short-History Cache

## Question

Does a short history of previous routing states improve cache performance compared with the first-order Markov policy?

The first-order Markov policy uses the current routing state.

The short-history policy also uses earlier routing states.

The test checks whether this additional history captures more of the LRU-to-oracle gap.

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

## Policies

The test compared four policies.

### LRU

LRU means Least Recently Used.

The cache keeps experts that were used most recently.

### First-order Markov

The first-order Markov policy uses learned expert transitions from the current routing state.

It uses only past routing observations.

It does not use future routing information.

### Short-history Markov

The short-history policy uses up to three recent routing states.

The most recent routing state has the highest weight.

Earlier routing states have lower weights.

The policy uses these weights:

* current routing state: 1.0;
* previous routing state: 0.5;
* routing state two steps earlier: 0.25.

The policy uses only past routing observations.

It does not use future routing information.

### Oracle

The oracle knows future expert requests.

The oracle is not a practical cache policy.

It provides an upper reference for the current simulation model.

## Cache capacities

The test used these cache capacities for each layer:

* 8 experts;
* 16 experts;
* 32 experts.

## Results

### Cache capacity: 8 experts

| Policy               | Hit rate |    Hits |  Misses |
| -------------------- | -------: | ------: | ------: |
| LRU                  |    42.4% | 524,883 | 713,517 |
| First-order Markov   |    47.6% | 588,916 | 649,484 |
| Short-history Markov |    47.3% | 585,261 | 653,139 |
| Oracle               |    60.8% | 753,417 | 484,983 |

The first-order Markov gain over LRU was:

**5.2 percentage points**

The short-history Markov gain over LRU was:

**4.9 percentage points**

The short-history policy performed:

**0.3 percentage points worse**

than the first-order Markov policy.

The short-history policy captured:

**26.4%**

of the available LRU-to-oracle gap.

### Cache capacity: 16 experts

| Policy               | Hit rate |    Hits |  Misses |
| -------------------- | -------: | ------: | ------: |
| LRU                  |    64.7% | 800,857 | 437,543 |
| First-order Markov   |    66.2% | 819,263 | 419,137 |
| Short-history Markov |    66.3% | 820,995 | 417,405 |
| Oracle               |    78.4% | 971,367 | 267,033 |

The short-history policy improved on the first-order Markov policy by:

**0.1 percentage points**

The short-history policy captured:

**11.8%**

of the available LRU-to-oracle gap.

### Cache capacity: 32 experts

| Policy               | Hit rate |      Hits |  Misses |
| -------------------- | -------: | --------: | ------: |
| LRU                  |    81.0% | 1,003,095 | 235,305 |
| First-order Markov   |    82.0% | 1,015,680 | 222,720 |
| Short-history Markov |    82.2% | 1,018,130 | 220,270 |
| Oracle               |    88.8% | 1,099,573 | 138,827 |

The short-history policy improved on the first-order Markov policy by:

**0.2 percentage points**

The short-history policy captured:

**15.6%**

of the available LRU-to-oracle gap.

## Main result

The short-history policy did not produce a material improvement over the first-order Markov policy.

At a capacity of 8 experts, the short-history policy performed slightly worse.

At capacities of 16 and 32 experts, the improvement was small.

The additional routing history therefore did not explain much more of the oracle headroom.

## Interpretation

The first-order Markov result from Finding 008 remains the stronger simple transition baseline.

The current short-history method does not justify more hand-built variants of the same approach.

The missing oracle headroom can depend on information that is not captured by a weighted combination of the last three routing states.

Possible additional information includes:

* router scores;
* layer identity;
* workload type;
* recent expert frequency;
* interactions between multiple features.

A learned predictor can test whether these features contain additional useful information.

## Practical meaning

The negative result is useful.

It removes one possible optimization direction.

The project does not need to test many additional history lengths or decay values before it tests a richer predictor.

This reduces unnecessary complexity.

The result also shows that more temporal context does not automatically produce a better cache policy.

## Important limitation

The test used one short-history design.

It used:

* three routing states;
* fixed decay weights;
* first-order expert transition counts.

The result does not prove that all history-based policies are ineffective.

It shows only that this simple short-history extension does not materially improve the current Markov baseline.

## Decision

**STOP HAND-BUILT SHORT-HISTORY VARIANTS**

The decision rule was:

* Continue with hand-built history methods if the short-history policy produces a clear improvement over first-order Markov.
* Stop this branch if the improvement is small or negative.

The measured difference was:

* -0.3 percentage points at capacity 8;
* +0.1 percentage points at capacity 16;
* +0.2 percentage points at capacity 32.

These differences are too small to justify further manual history variants.

## Next question

The next research question is:

> Can a lightweight learned predictor use richer routing features to capture more of the LRU-to-oracle gap?

The next test should use a simple learned model before a complex sequence model.

XGBoost is an appropriate next baseline.

If XGBoost does not materially improve on the first-order Markov policy, reconsider further predictor development.

If XGBoost captures substantially more of the oracle gap, continue to richer predictive methods.

## Status

Project stage: Predictive cache simulation

Finding: Short routing history does not materially improve on first-order Markov

Outcome: Stop this branch and test a learned predictor

