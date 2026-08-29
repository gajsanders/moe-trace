# Finding 012 — Minimal XGBoost Policy

## Question

Can a small XGBoost feature set reproduce the performance of the full 16-feature XGBoost model?

The objective is to reduce model complexity.

The test compares small feature sets with the full model.

## Data

The experiment used the same routing dataset and prompt split as Findings 010 and 011.

Training used:

* 20 prompts.

Testing used:

* 5 held-out prompts.

The held-out prompts were not used during training.

The training matrix contained:

`1,966,080`

rows.

The experiment tested cache capacities of:

* 8 experts per layer;
* 16 experts per layer.

## Models

The experiment compared these learned models:

* Markov score only;
* Markov score plus router scores;
* Markov score plus workload;
* Markov score plus router scores and workload;
* full 16-feature XGBoost.

The router-score model used:

* Markov score;
* current router score;
* previous router score.

The full model used 16 features.

## Reference policies

The experiment also reported:

* LRU;
* first-order Markov;
* Oracle.

LRU and Markov are practical baseline policies.

Oracle is an offline reference.

Oracle uses future routing information.

Oracle is not deployable.

## Results

### Cache capacity: 8 experts

| Policy or model                            | Hit rate |
| ------------------------------------------ | -------: |
| LRU                                        |   43.19% |
| Markov                                     |   47.41% |
| XGBoost: Markov only                       |   47.67% |
| XGBoost: Markov + workload                 |   47.67% |
| XGBoost: Markov + router scores            |   48.42% |
| XGBoost: Markov + router scores + workload |   48.43% |
| Full XGBoost                               |   48.47% |
| Oracle                                     |   61.26% |

The Markov-only XGBoost model improved on the Markov baseline by:

**0.26 percentage points**

The Markov-plus-router model improved on the Markov baseline by:

**1.01 percentage points**

The Markov-plus-router model was below the full XGBoost model by:

**0.05 percentage points**

Adding workload information to the Markov-plus-router model changed the result by:

**0.01 percentage points**

### Cache capacity: 16 experts

| Policy or model                            | Hit rate |
| ------------------------------------------ | -------: |
| LRU                                        |   65.05% |
| Markov                                     |   65.91% |
| XGBoost: Markov + workload                 |   66.31% |
| XGBoost: Markov only                       |   66.32% |
| Full XGBoost                               |   67.23% |
| XGBoost: Markov + router scores            |   67.23% |
| XGBoost: Markov + router scores + workload |   67.26% |
| Oracle                                     |   78.52% |

The Markov-only XGBoost model improved on the Markov baseline by:

**0.41 percentage points**

The Markov-plus-router model improved on the Markov baseline by:

**1.32 percentage points**

The Markov-plus-router model matched the full XGBoost result.

Adding workload information improved the result by only:

**0.03 percentage points**

## Main result

The Markov-plus-router model reproduced almost all of the full XGBoost performance.

It used only three features:

* Markov score;
* current router score;
* previous router score.

At capacity 8, the difference between the three-feature model and the full model was:

**0.05 percentage points**

At capacity 16, there was no measured difference.

The additional 13 features in the full model therefore provided little practical value in this experiment.

## Interpretation

The first-order Markov score contains the main temporal routing signal.

The router scores provide an additional useful signal.

The router scores improve performance more than workload information.

Workload information provides almost no additional value.

The larger feature set is not required to reproduce the full XGBoost result.

This result is consistent with Finding 011.

Finding 011 showed that removing the Markov score caused the largest performance loss.

Finding 012 shows that the Markov score plus router scores are sufficient to recover almost all of the full-model performance.

## Practical meaning

A small predictor is preferable if it provides the same cache performance as a larger predictor.

A smaller predictor can reduce:

* prediction cost;
* implementation complexity;
* memory use;
* maintenance cost.

These properties are important for an inference optimization.

The predictor must not cost more than the cache misses that it prevents.

The current result therefore supports a simple learned cache policy.

## Current policy candidate

The current learned-policy candidate uses:

* Markov transition score;
* current router score;
* previous router score.

The predictor uses XGBoost.

The current evidence does not justify a GRU or another complex sequence model.

## Important limitation

The result is based on one MoE model and one routing dataset.

The test set contains five held-out prompts.

The experiment measures simulated cache hit rate.

It does not measure inference speed.

It does not measure the real cost of an expert cache miss.

The result therefore identifies a candidate cache policy.

It does not prove a runtime speed improvement.

## Decision

**SELECT MARKOV PLUS ROUTER SCORES AS THE CURRENT LEARNED POLICY CANDIDATE**

**STOP MODEL COMPLEXITY WORK**

Do not start a GRU at this stage.

The three-feature model reproduces almost all of the full XGBoost result.

Additional model complexity is not justified by the current evidence.

## Next question

The next research question is:

> What is the real cost of an expert cache miss in the MLX Qwen3 execution path?

The next stage should move from predictive-model comparison to runtime investigation.

The project must determine whether a simulated cache miss corresponds to a meaningful runtime cost.

If cache misses are expensive, the selected policy can be tested in a real execution path.

If cache misses are cheap, further cache-prediction work can have little practical value.

## Status

Project stage: Learned cache policy selection

Finding: Markov score plus router scores reproduces the full XGBoost result

Outcome: Select the three-feature policy and move to runtime investigation
