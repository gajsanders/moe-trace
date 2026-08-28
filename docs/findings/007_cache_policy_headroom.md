# Finding 007 — Cache Policy Headroom

## Question

Does the observed routing structure create enough cache-performance headroom to justify a smarter cache policy?

The test compares simple cache policies with a future-aware oracle policy.

The oracle knows the future expert requests.

It therefore provides an upper reference for the current simulation model.

The main comparison is:

* LRU cache performance;
* oracle cache performance.

If LRU performs close to the oracle, a predictive cache has little useful headroom.

If the oracle performs substantially better than LRU, a predictive cache can possibly improve performance.

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

## Cache policies

The simulation tested these policies:

### Random

The cache removes experts at random when the cache exceeds its capacity.

This is a weak baseline.

### LFU

LFU means Least Frequently Used.

The cache keeps experts that have been requested most often.

### LRU

LRU means Least Recently Used.

The cache keeps experts that were used most recently.

LRU is the main practical baseline.

### Oracle

The oracle has access to future routing requests.

When the cache is full, it keeps the experts that will be needed soonest.

The oracle is not a practical runtime policy.

It provides an estimate of the maximum useful headroom in the current cache model.

## Cache capacities

The simulation tested these cache capacities for each layer:

* 8 experts;
* 16 experts;
* 32 experts;
* 64 experts.

Each MoE layer has 128 available experts.

Each routing decision selects 8 experts.

## Results

### Cache capacity: 8 experts

| Policy | Hit rate |    Hits |  Misses |
| ------ | -------: | ------: | ------: |
| Random |    36.5% | 451,762 | 786,638 |
| LFU    |    42.5% | 526,812 | 711,588 |
| LRU    |    42.4% | 524,883 | 713,517 |
| Oracle |    60.8% | 753,417 | 484,983 |

The LRU-to-oracle gap was:

**18.5 percentage points**

The oracle avoided:

**228,534**

misses compared with LRU.

### Cache capacity: 16 experts

| Policy | Hit rate |    Hits |  Misses |
| ------ | -------: | ------: | ------: |
| Random |    58.2% | 720,812 | 517,588 |
| LFU    |    61.5% | 761,087 | 477,313 |
| LRU    |    64.7% | 800,857 | 437,543 |
| Oracle |    78.4% | 971,367 | 267,033 |

The LRU-to-oracle gap was:

**13.8 percentage points**

The oracle avoided:

**170,510**

misses compared with LRU.

### Cache capacity: 32 experts

| Policy | Hit rate |      Hits |  Misses |
| ------ | -------: | --------: | ------: |
| Random |    76.7% |   949,273 | 289,127 |
| LFU    |    79.5% |   984,001 | 254,399 |
| LRU    |    81.0% | 1,003,095 | 235,305 |
| Oracle |    88.8% | 1,099,573 | 138,827 |

The LRU-to-oracle gap was:

**7.8 percentage points**

The oracle avoided:

**96,478**

misses compared with LRU.

### Cache capacity: 64 experts

| Policy | Hit rate |      Hits |  Misses |
| ------ | -------: | --------: | ------: |
| Random |    90.6% | 1,121,495 | 116,905 |
| LFU    |    91.7% | 1,135,080 | 103,320 |
| LRU    |    91.8% | 1,137,132 | 101,268 |
| Oracle |    92.9% | 1,150,530 |  87,870 |

The LRU-to-oracle gap was:

**1.1 percentage points**

The oracle avoided:

**13,398**

misses compared with LRU.

## Main result

The available cache-performance headroom depends strongly on cache capacity.

The largest gaps occurred with small caches.

| Cache capacity | LRU hit rate | Oracle hit rate |     Gap |
| -------------: | -----------: | --------------: | ------: |
|              8 |        42.4% |           60.8% | 18.5 pp |
|             16 |        64.7% |           78.4% | 13.8 pp |
|             32 |        81.0% |           88.8% |  7.8 pp |
|             64 |        91.8% |           92.9% |  1.1 pp |

The results show substantial theoretical headroom at capacities of 8 and 16 experts per layer.

The headroom decreases as cache capacity increases.

At 64 experts per layer, LRU performs close to the oracle.

## Interpretation

The routing structure contains information that LRU does not fully use.

This effect is strongest when the cache is small.

The result supports further tests of predictive or transition-aware cache policies.

The result also shows that simple expert popularity is not sufficient.

At a capacity of 16 experts:

* LFU achieved 61.5%;
* LRU achieved 64.7%;
* the oracle achieved 78.4%.

Thus, the oracle advantage is not explained by frequency alone.

Future routing order is important in the current simulation.

## Practical meaning

The result does not show that a real inference runtime will become faster by the same amount.

A cache miss in the simulation has one unit of cost.

The simulation does not yet model:

* expert size;
* SSD access time;
* unified-memory access;
* transfer latency;
* prefetch cost;
* prediction cost;
* asynchronous execution;
* hardware contention.

Therefore, the LRU-to-oracle gap is cache headroom.

It is not an expected inference-speed improvement.

## Important limitation

The oracle knows future requests.

A real cache policy does not have this information.

The oracle result shows only that better decisions are theoretically possible.

A practical policy must predict future requests with low computational cost.

The simulation also gives each layer an independent cache of the same size.

A real system can have one shared memory budget.

A later experiment can test unequal cache allocation between layers.

## Decision

**CONTINUE**

The decision rule was:

* Stop the predictive-cache investigation if LRU performs close to the oracle.
* Continue if the oracle has a substantial advantage over LRU.

The LRU-to-oracle gaps were:

* 18.5 percentage points at capacity 8;
* 13.8 percentage points at capacity 16;
* 7.8 percentage points at capacity 32;
* 1.1 percentage points at capacity 64.

The result meets the continuation condition for small and medium cache capacities.

The next tests should focus first on capacities of 8 and 16 experts per layer.

## Next question

The next research question is:

> How much of the LRU-to-oracle gap can a low-cost predictive policy capture?

The next stage should test simple methods before learned models.

The first candidate methods are:

* transition-aware cache policy;
* layer-aware transition policy;
* short-history policy.

A learned model should be tested only if simple methods leave useful headroom.

## Success metric

A useful metric for the next stage is:

$$
\text{gap captured}
=
\frac{
\text{hit rate}_{policy}
-
\text{hit rate}_{LRU}
}{
\text{hit rate}_{oracle}
-
\text{hit rate}_{LRU}
}
$$

This metric measures how much of the available LRU-to-oracle improvement a practical policy captures.

## Status

Project stage: Cache simulation

Finding: Substantial cache-policy headroom exists under tight cache limits

Outcome: Continue

