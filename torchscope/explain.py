"""Plain-language explanations of each layer, shown as captions in the viewer."""

from __future__ import annotations

import math

from torch import nn

from torchscope.graph import Graph, Node

ACTIVATIONS = {
    "ReLU": "ReLU then keeps positive values and sets negative ones to 0.",
    "relu": "ReLU then keeps positive values and sets negative ones to 0.",
    "ReLU6": "ReLU6 then clips values to the range 0 to 6.",
    "LeakyReLU": "LeakyReLU then keeps positive values and shrinks negative ones.",
    "GELU": "GELU (a smooth version of ReLU) then mostly keeps positive values.",
    "gelu": "GELU (a smooth version of ReLU) then mostly keeps positive values.",
    "SiLU": "SiLU (a smooth version of ReLU) then mostly keeps positive values.",
    "silu": "SiLU (a smooth version of ReLU) then mostly keeps positive values.",
    "Sigmoid": "Sigmoid then squashes each value into the range 0 to 1.",
    "sigmoid": "Sigmoid then squashes each value into the range 0 to 1.",
    "Tanh": "Tanh then squashes each value into the range -1 to 1.",
    "tanh": "Tanh then squashes each value into the range -1 to 1.",
    "Softmax": "Softmax then turns the values into probabilities that add up to 1.",
    "softmax": "Softmax then turns the values into probabilities that add up to 1.",
    "BatchNorm1d": "BatchNorm rescales each value to a stable range.",
    "BatchNorm2d": "BatchNorm rescales each channel to a stable range.",
    "LayerNorm": "LayerNorm rescales the values to a stable range.",
}

ACTIVATION_NAMES = {k for k in ACTIVATIONS if not k.startswith(("Batch", "Layer"))}


def _shape(shape: tuple[int, ...]) -> str:
    return "×".join(str(s) for s in shape) if shape else "a single number"


def _window(v: int | tuple[int, ...]) -> str:
    """Kernel or window size as ``"3×3"``."""
    v = v if isinstance(v, tuple) else (v, v)
    return "×".join(str(x) for x in v)


def _tags(node: Node) -> str:
    return " ".join(ACTIVATIONS[t] for t in node.tags if t in ACTIVATIONS)


def describe_input(node: Node) -> str:
    """Caption for an input node."""
    shape = node.shape[1:]
    if len(shape) == 3 and shape[0] in (1, 3):
        kind = "grayscale" if shape[0] == 1 else "color"
        return (
            f"The input: a {shape[1]}×{shape[2]} {kind} image, "
            f"{math.prod(shape)} numbers in total. Brightness shows each pixel's value."
        )
    if len(shape) == 2 and min(shape) >= 8:
        return f"The input: a {_shape(shape)} grid of numbers, drawn as an image."
    return f"The input: {_shape(shape)} numbers."


def describe(node: Node, graph: Graph) -> str:
    """Caption for one node of the graph, using its real sizes."""
    if node.role == "input":
        return describe_input(node)
    m = node.module
    out = node.shape[1:]
    prev = graph.nodes[node.inputs[0]].shape[1:] if node.inputs else ()
    tags = _tags(node)
    text = ""
    if isinstance(m, nn.Conv2d):
        text = (
            f"Convolution: {m.out_channels} filters, each {_window(m.kernel_size)} (×{m.in_channels} input "
            f"channel{'s' if m.in_channels > 1 else ''}), slide over the input. Each filter "
            f"produces one {out[1]}×{out[2]} feature map that lights up where its pattern appears."
        )
        stride = m.stride[0] if isinstance(m.stride, tuple) else m.stride
        if stride > 1:
            text += f" A stride of {stride} skips positions, which shrinks the maps."
    elif isinstance(m, nn.Linear):
        text = (
            f"A dense layer: each of the {m.out_features} outputs is a weighted sum of all "
            f"{m.in_features} inputs plus a bias. Positive weights add, negative weights subtract."
        )
    elif isinstance(m, (nn.MaxPool2d, nn.MaxPool1d)):
        text = (
            f"Max pooling keeps the largest value in each {_window(m.kernel_size)} "
            f"window, shrinking each map from {_shape(prev[1:])} to {_shape(out[1:])}."
        )
    elif isinstance(m, (nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
        text = f"Average pooling averages each window, shrinking each map from {_shape(prev[1:])} to {_shape(out[1:])}."
    elif isinstance(m, nn.MultiheadAttention):
        text = (
            "Self-attention: every token looks at every other token and mixes in "
            "information from the ones most relevant to it."
        )
    elif isinstance(m, nn.Embedding):
        text = (
            f"An embedding looks up a learned vector of {m.embedding_dim} numbers for each token."
        )
    elif isinstance(m, nn.RNNBase):
        text = f"A recurrent layer ({type(m).__name__}) reads the sequence step by step, keeping a memory of {m.hidden_size} numbers."
    elif node.role == "reshape":
        if len(out) == 1 and len(prev) > 1:
            text = (
                f"Flatten unrolls the {_shape(prev)} block into one list of {out[0]} numbers, "
                "so dense layers can use it."
            )
        else:
            text = f"Reshape ({node.type}): rearranges {_shape(prev)} into {_shape(out)} without changing any values."
    elif node.role == "merge":
        if node.type == "add":
            text = "Adds the incoming tensors element by element. When one of them skipped ahead from earlier, this is a skip (residual) connection."
        elif node.type == "cat":
            text = "Concatenates the incoming tensors, stacking their channels together."
        elif node.type == "mul":
            text = "Multiplies the incoming tensors element by element."
        else:
            text = f"Combines {len(node.inputs)} tensors with {node.type}."
    elif node.role == "norm":
        text = ACTIVATIONS.get(node.type, f"{node.type} rescales the values to a stable range.")
    elif node.role == "activation":
        text = ACTIVATIONS.get(node.type, f"{node.type} is applied to every value.")
    else:
        text = f"{node.type}: produces {_shape(out)}."
    if tags and node.role not in ("activation", "norm"):
        text += " " + tags
    if node.is_output and len(out) == 1 and out[0] > 1:
        text += f" These {out[0]} outputs are the model's scores, one per class; the highest is the prediction."
    return text


def describe_neuron_layer(
    k: int, n_layers: int, node: Node | None, size: int, shown: int, image: bool
) -> str:
    """Caption for one column of the neuron view."""
    hidden = f" Only the {shown} most active are drawn." if size > shown else ""
    if k == 0:
        source = "pixel" if image else "input"
        return (
            f"The input layer: each of the {size} {source} values becomes one neuron, "
            f"and its brightness is the value.{hidden}"
        )
    m = node.module
    act = next((t for t in node.tags if t in ACTIVATION_NAMES), None)
    text = (
        f"Each of these {m.out_features} neurons adds up all {m.in_features} neurons before it, "
        "each multiplied by its weight (blue adds, red subtracts), plus a bias."
    )
    if act:
        text += " " + ACTIVATIONS[act]
    if k == n_layers - 1:
        text += " These are the outputs: softmax turns them into probabilities, and the brightest is the prediction."
    return text + hidden
