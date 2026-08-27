from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock


@dataclass
class RoutingEvent:
    phase: str
    token_index: int
    layer: int
    expert_ids: list[int]
    scores: list[float]


@dataclass
class RoutingTrace:
    events: list[RoutingEvent] = field(default_factory=list)

    def clear(self) -> None:
        self.events.clear()


@contextmanager
def trace_qwen3_moe(model: Any):
    """
    Temporarily trace expert routing in Qwen3 MoE layers.

    Produces one RoutingEvent per token per MoE layer.
    Restores the original MLX-LM implementation afterwards.
    """

    trace = RoutingTrace()

    # Map MoE block instances to transformer layer numbers.
    layer_lookup = {}

    for layer_index, layer in enumerate(model.model.layers):
        moe = getattr(layer, "mlp", None)

        if isinstance(moe, Qwen3MoeSparseMoeBlock):
            layer_lookup[id(moe)] = layer_index

    # Track token positions separately for each layer.
    next_token_index = {
        layer_index: 0 for layer_index in layer_lookup.values()
    }

    original_call = Qwen3MoeSparseMoeBlock.__call__

    def traced_call(self, x):
        gates = self.gate(x)
        gates = mx.softmax(gates, axis=-1, precise=True)

        k = self.top_k

        inds = mx.argpartition(
            gates,
            kth=-k,
            axis=-1,
        )[..., -k:]

        scores = mx.take_along_axis(
            gates,
            inds,
            axis=-1,
        )

        if self.norm_topk_prob:
            scores /= mx.sum(scores, axis=-1, keepdims=True)

        # Preserve the original forward computation.
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        # Materialise routing data before converting to Python.
        mx.eval(inds, scores)

        expert_data = inds.tolist()
        score_data = scores.tolist()

        layer_index = layer_lookup.get(id(self), -1)

        # Expected shape for normal generation:
        # [batch, sequence_length, top_k]
        if len(expert_data) != 1:
            raise RuntimeError(
                f"MoE Trace currently supports batch size 1; "
                f"received batch size {len(expert_data)}"
            )

        expert_tokens = expert_data[0]
        score_tokens = score_data[0]

        sequence_length = len(expert_tokens)

        # A multi-token call is prompt prefill.
        # Single-token calls after that are autoregressive decoding.
        phase = "prefill" if sequence_length > 1 else "decode"

        if phase == "prefill":
            start_index = 0
            next_token_index[layer_index] = sequence_length
        else:
            start_index = next_token_index[layer_index]

        for offset, (token_experts, token_scores) in enumerate(
            zip(expert_tokens, score_tokens)
        ):
            trace.events.append(
                RoutingEvent(
                    phase=phase,
                    token_index=start_index + offset,
                    layer=layer_index,
                    expert_ids=token_experts,
                    scores=token_scores,
                )
            )

        if phase == "decode":
            next_token_index[layer_index] += sequence_length

        return y

    Qwen3MoeSparseMoeBlock.__call__ = traced_call

    try:
        yield trace
    finally:
        Qwen3MoeSparseMoeBlock.__call__ = original_call