# Finding 003 — Routing Trace Validation

## Question

Can MoE Trace produce a complete and consistent routing record for each token and each MoE layer?

The raw routing trace contained nested data.

This format was difficult to analyze.

The trace therefore needed a normalized structure.

## Normalized event format

MoE Trace was changed to create one routing event for each token and each MoE layer.

Each event contains:

- phase;
- token index;
- layer index;
- selected expert identifiers;
- routing scores.

The trace uses two phase values:

- `prefill`;
- `decode`.

Example:

```json
{
  "phase": "prefill",
  "token_index": 0,
  "layer": 0,
  "expert_ids": [113, 64, 47, 83, 56, 34, 63, 21],
  "scores": [
    0.060791015625,
    0.06298828125,
    0.07470703125,
    0.07568359375,
    0.08447265625,
    0.1376953125,
    0.1796875,
    0.322265625
  ]
}

The trace is written to a JSONL file.

Validation test

The validation test checked:

total number of events;
number of layers;
number of token positions;
number of events for each token;
number of events for each layer;
phase counts;
duplicate token-layer pairs.
Result

The trace contained:

Measurement	Result
Total events	1,584
MoE layers	48
Token positions	33
Expected events	1,584
Events per token	48
Events per layer	33
Prefill events	576
Decode events	1,008
Duplicate or missing token-layer pairs	0

The layer range was:

0 to 47

The token range was:

0 to 32

The expected rectangular trace size was:

48 × 33 = 1,584

The measured value was also:

1,584

Interpretation

The normalized trace passed the structural validation.

Each token has one routing event for each of the 48 MoE layers.

Each layer has one routing event for each of the 33 token positions.

No token-layer pair is missing.

No token-layer pair is duplicated.

The trace is therefore suitable for the first routing analyses.

This validation does not prove that all future traces will be correct.

Future work should add automated tests for the trace schema and indexing rules.

Decision

CONTINUE

The validation requirement was:

Produce a complete token-by-layer routing dataset without missing or duplicate records.

The trace met this requirement.

Status

Project stage: Trace normalization and validation

Outcome: Pass

Next stage: Routing characterization