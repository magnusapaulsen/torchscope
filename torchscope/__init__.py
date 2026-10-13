"""TorchScope: see what a PyTorch network looks like and watch data flow through it."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from torch import nn

from torchscope.export import build_payload
from torchscope.graph import Graph, Node
from torchscope.render import Visualization
from torchscope.trace import trace

__version__ = "0.1.0"

__all__ = ["Graph", "Node", "Visualization", "trace", "visualize", "__version__"]


def _first_samples(inputs: torch.Tensor | tuple[Any, ...], n: int) -> torch.Tensor | tuple:
    if isinstance(inputs, torch.Tensor):
        return inputs[:n]
    batch = next((t.shape[0] for t in inputs if isinstance(t, torch.Tensor) and t.ndim), None)
    return tuple(
        t[:n] if isinstance(t, torch.Tensor) and t.ndim and t.shape[0] == batch else t
        for t in inputs
    )


def visualize(
    model: nn.Module,
    inputs: torch.Tensor | tuple[Any, ...],
    *,
    view: str = "auto",
    max_neurons: int = 24,
    max_samples: int = 8,
    labels: Sequence[str] | None = None,
    image_shape: Sequence[int] | None = None,
    title: str | None = None,
    atomic: tuple[type[nn.Module], ...] = (),
) -> Visualization:
    """Trace ``model`` on example inputs and build an interactive, animated view of it.

    Small MLPs get the neuron view (every neuron and weight). Everything else gets the
    3D slab view (one slab per layer, textured with its activations). Each sample in the
    batch can be selected in the viewer.

    Args:
        model: Any ``nn.Module``. It is left unchanged.
        inputs: A batch tensor, or a tuple of positional arguments for ``model``.
        view: ``"auto"``, ``"neurons"`` or ``"slabs"``.
        max_neurons: Most neurons drawn per layer in the neuron view. Wider layers show
            their most active neurons.
        max_samples: Only the first ``max_samples`` samples of the batch are used.
        labels: Class names for the output layer.
        image_shape: ``(C, H, W)`` or ``(H, W)`` to draw a flat input as an image.
        title: Page title. Defaults to the model's class name.
        atomic: Extra module types to draw as one node instead of their children.

    Returns:
        A ``Visualization``. Call ``.save("net.html")`` or display it in Jupyter.

    Example:
        >>> viz = torchscope.visualize(model, images[:4], labels=class_names)
        >>> viz.save("model.html")
    """
    graph = trace(model, _first_samples(inputs, max_samples), atomic=atomic)
    payload = build_payload(
        graph,
        model,
        view=view,
        max_neurons=max_neurons,
        labels=labels,
        image_shape=image_shape,
        title=title,
    )
    return Visualization(payload, graph)
