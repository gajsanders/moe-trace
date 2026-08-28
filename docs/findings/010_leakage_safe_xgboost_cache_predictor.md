# Finding 010 — Leakage-Safe XGBoost Cache Predictor

## Question

Can a lightweight learned predictor improve expert cache performance compared with LRU and first-order Markov?

Can it do this on prompts that were not used for training?

The test uses XGBoost.

The model predicts whether a candidate expert will be selected on the next token.

The main objective is to test whether richer routing features contain useful predictive information.

## Data

The experiment used the cross-workload routing dataset from Finding 006.

The dataset contained:

* 25 prompts;
* 5 workload types;
* 3,225 decode token positions;
* 191,568 routing events;
* 48 MoE layers;
* 128 available experts;
* 8 selected experts for each token and layer.

The experiment used a prompt-level train/test split.

Training used:

* 20 prompts.

Testing used:

* 5 held-out prompts.

The held-out test prompts were:

* `coding_05`;
* `general_05`;
* `math_05`;
* `planning_05`;
* `summary_05`.

Each workload type therefore had one held-out test prompt.

The test prompts were not used to train XGBoost.

## Training data

The training process produced:

`1,966,080`

training rows.

The model used:

`16`

input features.

The positive-label rate was:

`50.0%`

Positive examples were experts that appeared in the next routing set.

Negative examples were sampled from experts that did not appear in the next routing set.

## Input features

The XGBoost predictor used information that was available at the current routing step.

The feature set included:

* expert identity;
* layer identity;
* current expert membership;
* current router score;
* previous expert membership;
* previous router score;
* recent four-token expert use;
* recent eight-token expert use;
* time since last expert use;
* historical expert frequency;
* first-order Markov score;
* workload type.

The model did not use the next routing decision as an input.

The model did not use future routing events during test-time cache decisions.

## Policies

The held-out test set compared four policies.

### LRU

LRU means Least Recently Used.

The cache keeps experts that were used most recently.

### First-order Markov

The Markov policy uses expert-transition counts from routing events that have already occurred.

It does not use future routing events.

### XGBoost

The XGBoost policy predicts the probability that each candidate expert will be selected on the next token.

The cache keeps the candidates with the highest predicted probability.

The predictor was trained on the 20 training prompts.

The predictor was evaluated only on the 5 held-out prompts.

### Oracle

The oracle knows the future routing sequence.

It is not a practical policy.

It provides an upper reference for the current simulation model.

## Cache capacities

The experiment tested these cache capacities for each layer:

* 8 experts;
* 16 experts;
* 32 experts.

## Results

### Cache capacity: 8 experts

| Policy  | Hit rate |    Hits |  Misses |
| ------- | -------: | ------: | ------: |
| LRU     |    43.2% | 106,975 | 140,705 |
| Markov  |    47.4% | 117,416 | 130,264 |
| XGBoost |    48.5% | 120,039 | 127,641 |
| Oracle  |    61.3% | 151,722 |  95,958 |

The Markov gain over LRU was:

**4.2 percentage points**

The XGBoost gain over LRU was:

**5.3 percentage points**

The XGBoost gain over Markov was:

**1.1 percentage points**

The available LRU-to-oracle gap was:

**18.1 percentage points**

XGBoost captured:

**29.2%**

of this gap.

XGBoost avoided:

**13,064**

misses compared with LRU.

### Cache capacity: 16 experts

| Policy  | Hit rate |    Hits | Misses |
| ------- | -------: | ------: | -----: |
| LRU     |    65.0% | 161,104 | 86,576 |
| Markov  |    65.9% | 163,240 | 84,440 |
| XGBoost |    67.2% | 166,513 | 81,167 |
| Oracle  |    78.5% | 194,484 | 53,196 |

The Markov gain over LRU was:

**0.9 percentage points**

The XGBoost gain over LRU was:

**2.2 percentage points**

The XGBoost gain over Markov was:

**1.3 percentage points**

The available LRU-to-oracle gap was:

**13.5 percentage points**

XGBoost captured:

**16.2%**

of this gap.

XGBoost avoided:

**5,409**

misses compared with LRU.

### Cache capacity: 32 experts

| Policy  | Hit rate |    Hits | Misses |
| ------- | -------: | ------: | -----: |
| LRU     |    81.3% | 201,335 | 46,345 |
| Markov  |    82.1% | 203,370 | 44,310 |
| XGBoost |    83.1% | 205,725 | 41,955 |
| Oracle  |    89.0% | 220,423 | 27,257 |

The Markov gain over LRU was:

**0.8 percentage points**

The XGBoost gain over LRU was:

**1.8 percentage points**

The XGBoost gain over Markov was:

**1.0 percentage point**

The available LRU-to-oracle gap was:

**7.7 percentage points**

XGBoost captured:

**23.0%**

of this gap.

XGBoost avoided:

**4,390**

misses compared with LRU.

## Main result

XGBoost performed better than LRU and first-order Markov at all tested cache capacities.

The improvement over Markov was consistent but small.

The XGBoost gain over Markov was:

* 1.1 percentage points at capacity 8;
* 1.3 percentage points at capacity 16;
* 1.0 percentage point at capacity 32.

The result shows that the richer feature set contains useful information that is not fully captured by first-order expert transitions.

However, most of the oracle headroom remains uncaptured.

## Interpretation

The learned predictor generalised to prompts that were not used for training.

This is important because the result is not based on scoring the same prompts that were used to fit the model.

The experiment therefore provides stronger evidence than an in-sample result.

The result also shows that XGBoost extracts additional predictive information beyond simple Markov transitions.

However, the additional improvement is modest.

The result does not yet justify a more complex sequence model.

The next test should identify which feature groups produce the XGBoost improvement.

## Practical meaning

XGBoost avoided more cache misses than LRU and Markov on the held-out test prompts.

This result can support a better cache-replacement policy.

However, the current experiment measures cache hit rate only.

It does not measure real inference speed.

The result does not show that inference will improve by the same percentage.

A real runtime test must include:

* model prediction cost;
* cache-management cost;
* memory or storage access cost;
* expert transfer cost;
* asynchronous execution;
* hardware contention.

## Important limitation

The held-out test set contains only 5 prompts.

Each workload type has one test prompt.

The result should therefore be treated as an initial generalisation test.

The predictor also uses a predefined feature set.

The current experiment does not show which features are necessary.

Some features can provide little useful information.

Some features can provide most of the measured improvement.

The next experiment must test this directly.

## Decision

**CONTINUE WITH FEATURE ABLATION**

The decision rule was:

* Reconsider learned prediction if XGBoost does not improve on Markov.
* Continue if XGBoost gives a consistent held-out improvement.
* Do not start a complex sequence model if the additional improvement is small.

XGBoost improved on Markov at all three tested cache capacities.

The improvement was consistent but limited to approximately 1.0 to 1.3 percentage points.

This result supports one additional diagnostic stage.

It does not yet support immediate development of a GRU or other complex sequence model.

## Next question

The next research question is:

> Which feature groups are responsible for the XGBoost improvement?

The next experiment should remove feature groups and retrain the model.

Candidate ablations include:

* remove workload features;
* remove router-score features;
* remove Markov score;
* remove recency and history features;
* remove expert identity;
* remove layer identity.

The experiment should compare each ablated model with the full XGBoost model.

If one or two simple feature groups explain most of the improvement, prefer a simpler predictor.

If multiple interacting features are necessary, a richer learned model can be justified.

## Status

Project stage: Learned predictive cache simulation

Finding: XGBoost gives a small but consistent improvement on held-out prompts

Outcome: Continue with feature ablation before a complex sequence model
