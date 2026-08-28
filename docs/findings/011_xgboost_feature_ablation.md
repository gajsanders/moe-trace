# Finding 011 — XGBoost Feature Ablation

## Question

Which feature groups are responsible for the XGBoost cache improvement?

The previous experiment showed that XGBoost performed better than LRU and first-order Markov on held-out prompts.

The improvement over Markov was small.

This experiment removes feature groups from XGBoost.

The objective is to identify which features produce the measured improvement.

## Data

The experiment used the same routing dataset and prompt split as Finding 010.

The dataset contained:

* 25 prompts;
* 5 workload types;
* 48 MoE layers;
* 128 available experts;
* 8 selected experts for each token and layer.

Training used:

* 20 prompts.

Testing used:

* 5 held-out prompts.

The held-out prompts were not used during model training.

The experiment used:

`1,966,080`

training rows.

The full model used:

`16`

features.

## Method

The full XGBoost model was used as the reference.

Each ablation removed one feature or one feature group.

The model was retrained after each removal.

All other experiment settings remained unchanged.

The experiment tested these ablations:

* remove expert identity;
* remove layer identity;
* remove router scores;
* remove Markov score;
* remove recency and history features;
* remove workload features.

The recency and history group contained:

* previous expert membership;
* recent four-token use;
* recent eight-token use;
* time since last use;
* historical expert frequency.

The router-score group contained:

* current router score;
* previous router score.

The workload group contained the five workload-type indicators.

## Cache capacities

The ablation study tested:

* 8 experts per layer;
* 16 experts per layer.

These capacities were selected because meaningful oracle headroom remained at these sizes.

## Reference results

### Cache capacity: 8 experts

The reference hit rates were:

| Policy       | Hit rate |
| ------------ | -------: |
| LRU          |   43.19% |
| Markov       |   47.41% |
| Full XGBoost |   48.47% |
| Oracle       |   61.26% |

### Cache capacity: 16 experts

The reference hit rates were:

| Policy       | Hit rate |
| ------------ | -------: |
| LRU          |   65.05% |
| Markov       |   65.91% |
| Full XGBoost |   67.23% |
| Oracle       |   78.52% |

## Results

### Cache capacity: 8 experts

| Ablation                   | Hit rate |  Change from full model |
| -------------------------- | -------: | ----------------------: |
| Full XGBoost               |   48.47% |                       — |
| Remove Markov score        |   47.82% | -0.64 percentage points |
| Remove workload            |   48.18% | -0.29 percentage points |
| Remove router scores       |   48.27% | -0.20 percentage points |
| Remove layer               |   48.30% | -0.16 percentage points |
| Remove expert identity     |   48.39% | -0.08 percentage points |
| Remove recency and history |   48.54% | +0.08 percentage points |

The largest performance loss occurred when the Markov score was removed.

Removing the recency and history group produced a small improvement.

### Cache capacity: 16 experts

| Ablation                   | Hit rate |  Change from full model |
| -------------------------- | -------: | ----------------------: |
| Full XGBoost               |   67.23% |                       — |
| Remove Markov score        |   66.19% | -1.04 percentage points |
| Remove recency and history |   67.14% | -0.09 percentage points |
| Remove expert identity     |   67.17% | -0.06 percentage points |
| Remove router scores       |   67.28% | +0.05 percentage points |
| Remove workload            |   67.29% | +0.06 percentage points |
| Remove layer               |   67.32% | +0.09 percentage points |

The Markov score again had the largest effect.

Several other feature removals slightly improved the result.

## Main result

The Markov score was the most important feature in the full XGBoost model.

At capacity 8, removing the Markov score reduced the hit rate by:

**0.64 percentage points**

At capacity 16, removing the Markov score reduced the hit rate by:

**1.04 percentage points**

No other feature group produced a similar loss.

Most other feature groups had a small effect.

Some feature removals slightly improved performance.

## Relationship to the Markov baseline

At capacity 8:

* Markov achieved 47.41%;
* XGBoost without the Markov score achieved 47.82%;
* full XGBoost achieved 48.47%.

The non-Markov features therefore produced only a small improvement above the standalone Markov policy.

At capacity 16:

* Markov achieved 65.91%;
* XGBoost without the Markov score achieved 66.19%;
* full XGBoost achieved 67.23%.

Again, most of the additional XGBoost performance depended on the Markov-derived feature.

## Interpretation

The full XGBoost model does not appear to depend on a large set of independent predictive signals.

The first-order Markov signal explains a large part of the learned-model improvement.

The remaining features provide only small additional gains.

The current router-score features do not show strong additional predictive value.

The workload features do not show strong additional predictive value.

Layer identity does not show strong additional predictive value.

Expert identity does not show strong additional predictive value.

The tested recency and history features also do not show strong additional predictive value.

This result is consistent with Finding 009.

Finding 009 showed that a hand-built short-history Markov policy did not materially improve on first-order Markov.

The current ablation shows that removing short-term history features from XGBoost also has little effect.

These two results suggest that the useful routing signal can be largely low-order.

## Practical meaning

The result reduces the justification for a complex temporal predictor.

A complex model can add runtime cost.

A complex model can also increase implementation and maintenance cost.

The current evidence does not show that this additional complexity is necessary.

A simpler predictor based on the Markov score can be more appropriate.

The next experiment should test whether a small feature set can reproduce most of the full XGBoost result.

## Important limitation

This experiment tests feature removal from one XGBoost configuration.

A small ablation effect does not prove that the removed information has no predictive value.

A different feature representation can use the same information more effectively.

A different model can also learn different interactions.

The current result applies only to the tested feature definitions and model configuration.

The test set also contains only five held-out prompts.

The result must therefore be treated as an initial diagnostic result.

## Decision

**DO NOT START A GRU YET**

The ablation result does not show strong evidence that rich temporal or contextual features are necessary.

The first-order Markov score explains most of the measured XGBoost improvement.

The next step should test minimal learned models.

## Next question

The next research question is:

> Can a small XGBoost feature set reproduce most of the full-model result?

Candidate models include:

* Markov score only;
* Markov score plus current router score;
* Markov score plus workload;
* Markov score plus router score and workload;
* full XGBoost reference model.

If a minimal model reproduces the full result, prefer the simpler predictor.

If the minimal models fail to reproduce the full result, reconsider whether feature interactions justify a richer model.

## Status

Project stage: Learned predictive cache analysis

Finding: The Markov score explains most of the XGBoost improvement

Outcome: Test minimal Markov-plus models before any complex sequence model
