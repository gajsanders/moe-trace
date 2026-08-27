from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx
from mlx_lm.models.qwen3_moe import Qwen3MoeSparseMoeBlock


@dataclass
class RoutingEvent:
    layer: int
    expert_ids: list
    scores: list


@dataclass
class RoutingTrace:
    events: list[RoutingEvent] = field(default_factory=list)

    def clear(self) -> None:
        self.events.clear()


@contextmanager
def trace_qwen3_moe(model: Any):
    """
    Temporarily trace expert routing in Qwen3 MoE layers.

    Restores the original MLX-LM implementation afterwards.
    """

    trace = RoutingTrace()

    # Map each MoE block instance to its transformer layer number.
    layer_lookup = {}

    for layer_index, layer in enumerate(model.model.layers):
        moe = getattr(layer, "mlp", None)

        if isinstance(moe, Qwen3MoeSparseMoeBlock):
            layer_lookup[id(moe)] = layer_index

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

        # Keep the actual forward computation identical to MLX-LM.
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)

        mx.eval(inds, scores)

        trace.events.append(
            RoutingEvent(
                layer=layer_lookup.get(id(self), -1),
                expert_ids=inds.tolist(),
                scores=scores.tolist(),
            )
        )

        return y

    Qwen3MoeSparseMoeBlock.__call__ = traced_call

    try:
        yield trace
    finally:
        Qwen3MoeSparseMoeBlock.__call__ = original_call