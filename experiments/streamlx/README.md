# streamlx Runtime Experiment

This directory preserves the streamlx integration used for MoE Trace
Finding 015.

The experiment tests predictive expert-cache eviction in a real
SSD-backed constrained-memory MoE runtime.

## Upstream runtime

streamlx:
https://github.com/srcterm/streamlx

The upstream runtime is not vendored into this repository.

`moe_trace_streamlx.patch` contains the experimental changes made to
`streamlx/pool.py` and `streamlx/integrate.py`.

## Benchmark

`benchmark_stage15d_heldout.py` runs the held-out 4 GiB runtime
comparison used in Finding 015.

The five held-out prompts are:

- coding_05
- general_05
- math_05
- planning_05
- summary_05

streamlx predictive prefetch is disabled.

## Predictor

The runtime predictor uses the three-feature model selected in
Finding 012:

- Markov score
- current router score
- previous router score

The generated XGBoost model file is not tracked.

Recreate it with:

    python scripts/train_runtime_markov_router.py

The model metadata is preserved in:

    artifacts/markov_router_metadata.json

## Results

`results/stage15d_lru_4g.json` contains the held-out LRU baseline.

`results/stage15d_markov_router_4g.json` contains the held-out
Markov + router result.

See:

    docs/findings/015_predictive_cache_runtime_validation.md

for the interpretation and stop decision.
