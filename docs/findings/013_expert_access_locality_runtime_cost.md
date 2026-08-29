# Finding 013 — Expert Access Locality Has a Measurable MLX Runtime Cost

## Question

Does expert access locality have a measurable runtime cost in MLX?

The objective is to connect the previous routing and cache simulations to the real MLX execution path.

The investigation had four parts:

* inspect expert parameter storage;
* inspect the expert execution path;
* measure expert access locality in one layer;
* repeat the measurement across multiple layers.

## Model

The investigation used:

`mlx-community/Qwen3-30B-A3B-4bit`

The model contains:

* 48 decoder layers;
* 48 MoE blocks;
* 128 experts per MoE layer;
* top-8 expert routing.

The model used the MLX GPU device.

## 13A — Expert storage layout

The first test inspected the stored expert parameters.

Each MoE layer contained a `SwitchGLU` block.

The expert weights were stored in packed tensors.

The first tensor dimension contained all 128 experts.

Each layer contained quantized expert tensors for:

* gate projection;
* up projection;
* down projection.

The measured storage was:

* 324.00 MiB per MoE layer;
* 15.19 GiB across all 48 MoE layers;
* approximately 2.53 MiB per expert.

The expert parameters therefore form a substantial part of the model.

## 13B — Expert execution path

The second test inspected the installed MLX-LM implementation.

`SwitchGLU` receives the expert indices selected by the router.

The model uses:

`QuantizedSwitchLinear`

for:

* `gate_proj`;
* `up_proj`;
* `down_proj`.

Each projection uses:

* 128 experts;
* 4-bit quantization;
* group size 64.

`QuantizedSwitchLinear` passes the selected expert indices to:

`mx.gather_qmm`

through:

`rhs_indices=indices`

The execution path is therefore:

```text
router
  ↓
top-8 expert indices
  ↓
SwitchGLU
  ↓
QuantizedSwitchLinear
  ↓
mx.gather_qmm
  ↓
selected expert weight slices
```

This confirms sparse expert computation.

MLX does not calculate all 128 experts for each token.

The selected expert indices directly control which expert weight slices are used.

## Existing locality optimization

`SwitchGLU` also contains an access-order optimization.

When the number of routed items is at least 64, MLX sorts the routing indices.

The source code states that this is done so that access to different experts occurs in order.

This shows that expert memory-access order is already relevant to the MLX implementation.

This optimization is not the same as a cross-token expert cache.

No explicit:

* LRU policy;
* expert eviction;
* expert prefetch;
* SSD load;
* cross-token residency manager

was identified in this execution path.

## 13C — Single-layer locality benchmark

The third test measured whether different expert access patterns changed `SwitchGLU` execution time.

The benchmark used layer 0.

Each call selected exactly eight experts.

The amount of expert computation remained constant.

Three access patterns were tested.

### same_8

The same eight experts were selected for every call.

### alternate_16

The benchmark alternated between two groups of eight experts.

The working set contained 16 experts.

### rotate_128

The benchmark rotated through 16 groups of eight experts.

The working set contained all 128 experts.

The benchmark used:

* 50 warm-up calls;
* 250 measured calls per round;
* 8 rounds;
* forced MLX evaluation after each call;
* randomized condition order between rounds.

The results were:

| Access pattern | Median latency | Difference from same_8 |
| -------------- | -------------: | ---------------------: |
| same_8         |       0.176 ms |               baseline |
| alternate_16   |       0.182 ms |                 +3.41% |
| rotate_128     |       0.186 ms |                 +5.66% |

The same-eight condition was consistently faster.

The rotate-128 condition was consistently slower.

This provided initial evidence of a resident-memory expert locality effect.

## 13D — Cross-layer replication

The fourth test repeated the benchmark across multiple MoE layers.

The selected layers were:

* layer 0;
* layer 7;
* layer 18;
* layer 31;
* layer 47.

These layers provide samples from different parts of the model.

The benchmark configuration remained unchanged.

## Cross-layer results

| Layer |   same_8 | alternate_16 | rotate_128 | Alternate penalty | Rotate penalty |
| ----: | -------: | -----------: | ---------: | ----------------: | -------------: |
|     0 | 0.171 ms |     0.182 ms |   0.185 ms |            +6.23% |         +8.15% |
|     7 | 0.172 ms |     0.180 ms |   0.185 ms |            +4.93% |         +8.02% |
|    18 | 0.168 ms |     0.178 ms |   0.185 ms |            +5.54% |         +9.70% |
|    31 | 0.170 ms |     0.179 ms |   0.184 ms |            +5.51% |         +8.48% |
|    47 | 0.171 ms |     0.180 ms |   0.186 ms |            +5.37% |         +8.45% |

Aggregate results:

* median `alternate_16` penalty: **+5.51%**;
* median `rotate_128` penalty: **+8.45%**;
* `alternate_16` was slower than `same_8` on **5 of 5 layers**;
* `rotate_128` was slower than `same_8` on **5 of 5 layers**.

## Main result

Expert access locality has a measurable runtime effect in MLX.

When the amount of expert computation is held constant, repeatedly accessing the same eight experts is faster than accessing a larger expert working set.

Rotating through all 128 experts produced a median latency penalty of:

**+8.45%**

across the five tested layers.

The effect was positive on all tested layers.

## Interpretation

This result is important because the model already fits in unified memory.

The benchmark did not require SSD loading or explicit expert offloading.

The locality effect therefore exists even when all expert parameters are available to the runtime.

A wider expert working set causes a measurable increase in `SwitchGLU` execution time.

The experiment does not identify the exact hardware mechanism.

Possible causes can include:

* processor cache behavior;
* memory-access locality;
* Metal kernel memory behavior;
* unified-memory access patterns;
* quantized gather behavior.

The benchmark only establishes the measured timing effect.

## Relationship to earlier findings

Earlier findings established that expert routing has temporal structure.

Adjacent tokens frequently reuse some experts.

The amount of reuse varies by layer and workload.

The cache simulations then showed that:

* LRU captures some reuse;
* Markov transition information improves over LRU;
* XGBoost adds a small additional improvement;
* a three-feature model using Markov and router scores reproduces almost all full XGBoost performance.

Finding 013 adds the missing runtime evidence.

The combined chain is now:

```text
expert routing has temporal locality
        ↓
future expert use is partly predictable
        ↓
simple prediction improves simulated locality
        ↓
expert access locality affects real MLX execution time
```

This establishes a plausible runtime path for the earlier cache-policy work.

## Practical meaning

The result does not show that the current predictive policy improves full-model inference speed.

It shows that the optimization target has a real runtime cost.

This distinction is important.

The previous cache simulations optimized expert locality.

Finding 013 shows that expert locality can affect real execution latency.

The next stage can therefore test whether a routing-aware policy can convert this effect into an end-to-end inference improvement.

## Important limitation

The measured percentage differences apply only to the isolated `SwitchGLU` benchmark.

They are not full-model speed improvements.

The benchmark does not include:

* attention;
* router calculation;
* normalization;
* KV-cache operations;
* token sampling;
* other MLX kernels;
* full model scheduling.

The result therefore must not be interpreted as:

“MoE Trace can make Qwen3 8.45% faster.”

The 8.45% value is the median penalty measured inside the isolated expert-locality benchmark.

The end-to-end effect can be smaller.

## Additional limitation

The benchmark used synthetic expert-selection patterns.

It did not replace the model router.

The conditions were designed to isolate expert-access locality.

They do not represent a complete natural inference sequence.

The next experiment must use real routing behavior.

## Decision

**CONTINUE**

The runtime investigation found a repeatable expert-locality effect.

The effect was present across all five tested layers.

This provides sufficient evidence to continue from simulation to a real runtime-policy experiment.

Do not develop a more complex prediction model at this stage.

The current Markov-plus-router approach remains the preferred candidate because it captures nearly all measured XGBoost benefit with a small feature set.

## Next question

The next research question is:

> Can a routing-aware locality policy produce a measurable end-to-end inference improvement on real Qwen3 decoding?

The next stage must connect:

* real router output;
* expert locality;
* policy overhead;
* full inference latency.

The test must compare the optimized path with an unchanged MLX-LM baseline.

A useful result requires a repeatable wall-clock improvement.

If the policy overhead removes the locality benefit, stop this optimization branch.

If a measurable end-to-end improvement remains, continue to broader model and hardware validation.

## Status

Project stage: Runtime optimization preparation

Finding: Expert access locality has a measurable MLX runtime cost

Cross-layer result: `rotate_128` had a median **+8.45%** `SwitchGLU` latency penalty compared with `same_8`

Replication: Positive effect on **5 of 5** tested layers

Next step: End-to-end routing-aware runtime experiment
