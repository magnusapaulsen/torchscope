"""Turn a traced graph into the JSON payload the viewer renders."""

from __future__ import annotations

import base64
import math
from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from torchscope.graph import Graph, Node

MAX_MLP_LAYERS = 12


def _b64(t: torch.Tensor) -> str:
    return base64.b64encode(bytes(t.flatten().to(torch.uint8).tolist())).decode("ascii")


def _quantize(x: torch.Tensor, per_sample: bool = False) -> tuple[list[str], float, float]:
    """Map a ``(B, ...)`` float tensor to uint8, robust to outliers. Returns data, lo, hi."""
    x = torch.nan_to_num(x.float())
    if per_sample:
        flat = x.flatten(1)
        lo = flat.min(dim=1).values.view(-1, *[1] * (x.ndim - 1))
        hi = flat.max(dim=1).values.view(-1, *[1] * (x.ndim - 1))
        q = ((x - lo) / (hi - lo).clamp_min(1e-12) * 255).round()
        return [_b64(s) for s in q], float(lo.min()), float(hi.max())
    flat = x.flatten()
    if flat.numel() > 1:
        lo, hi = torch.quantile(flat, torch.tensor([0.01, 0.99])).tolist()
    else:
        lo = hi = float(flat[0])
    if hi - lo < 1e-12:
        lo, hi = lo - 0.5, hi + 0.5
    q = ((x - lo) / (hi - lo)).clamp(0, 1).mul(255).round()
    return [_b64(s) for s in q], lo, hi


def _face(x: torch.Tensor) -> dict[str, Any]:
    data, lo, hi = _quantize(x)
    return {"h": x.shape[1], "w": x.shape[2], "lo": lo, "hi": hi, "data": data}


def _rounded(x: torch.Tensor) -> list:
    return [[round(v, 5) for v in row] for row in x.tolist()]


def _probabilities(values: torch.Tensor) -> torch.Tensor:
    """Softmax, unless the values already look like probabilities."""
    if values.min() >= 0 and torch.allclose(values.sum(-1), torch.ones(values.shape[0]), atol=1e-3):
        return values
    return values.softmax(dim=-1)


def _display_label(node: Node) -> str:
    """Last scope component, made readable for ``nn.Sequential`` indices."""
    if not node.scope:
        return node.name
    last = node.scope[-1]
    if last.split(":")[0].isdigit():
        return f"{node.scope[-2]}.{last}" if len(node.scope) > 1 else node.type
    return last


def is_mlp(graph: Graph) -> bool:
    """Whether ``graph`` is a single chain of ``nn.Linear`` layers on a flat input.

    Reshape nodes (``nn.Flatten``, ``view``) are allowed only before the first layer.
    """
    nodes = graph.nodes
    if not nodes or nodes[0].role != "input" or len(nodes) < 2:
        return False
    for i, n in enumerate(nodes[1:], start=1):
        if n.inputs != [i - 1]:
            return False
    body = nodes[1:]
    while body and body[0].role == "reshape":
        body = body[1:]
    if not body or len(body) > MAX_MLP_LAYERS:
        return False
    first_in = nodes[len(nodes) - len(body) - 1]
    if len(first_in.shape) != 2 or first_in.summary is None or first_in.summary.vector is None:
        return False
    return all(
        isinstance(n.module, nn.Linear)
        and len(n.shape) == 2
        and n.summary is not None
        and n.summary.vector is not None
        for n in body
    )


def _image_dims(
    input_node: Node, flat_size: int, image_shape: Sequence[int] | None
) -> tuple[int, int, int] | None:
    shape = tuple(image_shape) if image_shape else tuple(input_node.shape[1:])
    if len(shape) == 2:
        shape = (1, *shape)
    if len(shape) == 3 and shape[0] in (1, 3) and math.prod(shape) == flat_size:
        return shape  # type: ignore[return-value]
    return None


def neuron_payload(
    graph: Graph,
    max_neurons: int,
    labels: Sequence[str] | None,
    image_shape: Sequence[int] | None,
) -> dict[str, Any]:
    """Build the neuron-view payload for an MLP graph (see ``is_mlp``)."""
    nodes = graph.nodes
    body = [n for n in nodes if isinstance(n.module, nn.Linear)]
    first_in = nodes[nodes.index(body[0]) - 1]
    input_values = first_in.summary.vector  # (B, D0)

    def pick(values: torch.Tensor) -> list[int]:
        if values.shape[1] <= max_neurons:
            return list(range(values.shape[1]))
        top = values.abs().mean(dim=0).topk(max_neurons).indices
        return sorted(top.tolist())

    shown = pick(input_values)
    layers: list[dict[str, Any]] = [
        {
            "name": "input",
            "label": "Input",
            "detail": f"{input_values.shape[1]} values",
            "size": input_values.shape[1],
            "shown": shown,
            "values": _rounded(input_values[:, shown]),
        }
    ]
    for n in body:
        values = n.summary.vector
        prev_shown, shown = shown, pick(values)
        w = n.module.weight.detach().float().cpu()
        b = n.module.bias
        tags = " · ".join(n.tags)
        layers.append(
            {
                "name": n.name,
                "label": _display_label(n),
                "detail": f"Linear {w.shape[1]}→{w.shape[0]}" + (f" · {tags}" if tags else ""),
                "size": values.shape[1],
                "shown": shown,
                "values": _rounded(values[:, shown]),
                "weights": _rounded(w[shown][:, prev_shown]),
                "bias": [round(v, 5) for v in b.detach().float().cpu()[shown].tolist()]
                if b is not None
                else None,
            }
        )

    payload: dict[str, Any] = {"layers": layers, "image": None, "output": None}
    dims = _image_dims(nodes[0], input_values.shape[1], image_shape)
    if dims is not None:
        c, h, w = dims
        pixels = input_values.reshape(-1, c, h, w).permute(0, 2, 3, 1)  # interleaved HWC
        data, _, _ = _quantize(pixels)
        payload["image"] = {"c": c, "h": h, "w": w, "data": data}

    last = body[-1].summary.vector
    out_shown = layers[-1]["shown"]
    if len(out_shown) <= 32:
        probs = _probabilities(last)
        payload["output"] = {
            "labels": [str(labels[i]) if labels else str(i) for i in out_shown],
            "probs": _rounded(probs[:, out_shown]),
            "predicted": probs.argmax(dim=1).tolist(),
        }
    return payload


def graph_payload(graph: Graph, labels: Sequence[str] | None) -> dict[str, Any]:
    """Build the slab-view payload: nodes with face textures, plus edges."""
    out_nodes = []
    for n in graph.nodes:
        s = n.summary
        node: dict[str, Any] = {
            "id": n.id,
            "kind": n.kind,
            "role": n.role,
            "name": n.name,
            "label": _display_label(n),
            "type": n.type,
            "tags": n.tags,
            "scope": list(n.scope),
            "inputs": n.inputs,
            "shape": list(n.shape[1:]),
            "info": n.info,
            "params": n.params,
            "output": n.is_output,
        }
        if s is not None:
            node["dims"] = list(s.dims)
            node["faces"] = {"front": _face(s.front), "side": _face(s.side), "top": _face(s.top)}
            if s.image is not None:
                img = s.image.permute(0, 2, 3, 1)  # (B, h, w, C) interleaved
                data, _, _ = _quantize(img, per_sample=True)
                node["image"] = {
                    "c": img.shape[3],
                    "h": img.shape[1],
                    "w": img.shape[2],
                    "data": data,
                }
            if n.is_output and len(n.shape) == 2 and n.shape[1] >= 2 and s.vector is not None:
                probs = _probabilities(s.vector)
                k = min(5, probs.shape[1])
                top = probs.topk(k, dim=1)
                node["top"] = [
                    [
                        [str(labels[i]) if labels and i < len(labels) else str(i), round(p, 4)]
                        for i, p in zip(idx.tolist(), val.tolist(), strict=True)
                    ]
                    for idx, val in zip(top.indices, top.values, strict=True)
                ]
        out_nodes.append(node)
    return {"nodes": out_nodes}


def build_payload(
    graph: Graph,
    model: nn.Module,
    *,
    view: str = "auto",
    max_neurons: int = 24,
    labels: Sequence[str] | None = None,
    image_shape: Sequence[int] | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """Assemble the full viewer payload.

    Args:
        graph: A simplified graph from ``trace``.
        model: The traced model, for its name and parameter count.
        view: ``"auto"``, ``"neurons"`` or ``"slabs"``.
        max_neurons: Most neurons drawn per layer in the neuron view.
        labels: Class names for the output layer.
        image_shape: ``(C, H, W)`` or ``(H, W)`` to draw a flat input as an image.
        title: Page title. Defaults to the model's class name.

    Returns:
        A JSON-serializable dict.

    Raises:
        ValueError: If ``view="neurons"`` is requested for a model that is not an MLP.
    """
    if view not in ("auto", "neurons", "slabs"):
        raise ValueError(f"view must be 'auto', 'neurons' or 'slabs', got {view!r}.")
    mlp = is_mlp(graph)
    if view == "neurons" and not mlp:
        raise ValueError(
            "The neuron view needs a chain of nn.Linear layers on a flat input. Use view='slabs'."
        )
    chosen = "neurons" if (view == "neurons" or (view == "auto" and mlp)) else "slabs"
    name = type(model).__name__
    return {
        "version": 1,
        "title": title or name,
        "model": {
            "name": name,
            "params": sum(p.numel() for p in model.parameters()),
            "input_shape": list(graph.nodes[0].shape) if graph.nodes else [],
        },
        "samples": graph.nodes[0].shape[0] if graph.nodes else 0,
        "view": chosen,
        "neurons": neuron_payload(graph, max_neurons, labels, image_shape)
        if chosen == "neurons"
        else None,
        # The slab payload is always included, so the viewer can switch views.
        "graph": graph_payload(graph, labels),
    }
