"""Record a model's forward pass as a dataflow graph of module calls and tensor ops."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.overrides import TorchFunctionMode

from torchscope.graph import Graph, Node, simplify
from torchscope.summary import summarize

# Modules drawn as a single node even though they have children.
ATOMIC_TYPES: tuple[type[nn.Module], ...] = (nn.MultiheadAttention, nn.RNNBase)

ACTIVATION_TYPES: tuple[type[nn.Module], ...] = (
    nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU, nn.Sigmoid, nn.Tanh, nn.LeakyReLU, nn.ELU, nn.SELU,
    nn.CELU, nn.PReLU, nn.Softmax, nn.LogSoftmax, nn.Softplus, nn.Hardswish, nn.Hardsigmoid,
    nn.Hardtanh, nn.Mish,
)  # fmt: skip
NORM_TYPES: tuple[type[nn.Module], ...] = (
    nn.modules.batchnorm._BatchNorm, nn.modules.instancenorm._InstanceNorm, nn.LayerNorm,
    nn.GroupNorm, nn.LocalResponseNorm,
) + ((nn.RMSNorm,) if hasattr(nn, "RMSNorm") else ())  # fmt: skip
PASSTHROUGH_TYPES: tuple[type[nn.Module], ...] = (nn.modules.dropout._DropoutNd, nn.Identity)
RESHAPE_TYPES: tuple[type[nn.Module], ...] = (nn.Flatten, nn.Unflatten)

ACTIVATION_OPS = {
    "relu", "relu6", "gelu", "silu", "sigmoid", "tanh", "softmax", "log_softmax", "leaky_relu",
    "elu", "selu", "celu", "hardswish", "hardsigmoid", "hardtanh", "mish", "softplus",
}  # fmt: skip
NORM_OPS = {"layer_norm", "batch_norm", "group_norm", "instance_norm", "rms_norm", "normalize"}
# Shape-preserving unary ops that are not worth showing at all.
SILENT_OPS = {
    "contiguous", "clone", "detach", "to", "float", "double", "half", "bfloat16", "type",
    "type_as", "cpu", "cuda", "dropout", "dropout1d", "dropout2d", "dropout3d", "alpha_dropout",
    "feature_alpha_dropout", "requires_grad", "retain_grad", "view_as", "reshape_as",
}  # fmt: skip
_OP_ALIASES = {
    "iadd": "add", "radd": "add", "isub": "sub", "rsub": "sub", "imul": "mul", "rmul": "mul",
    "truediv": "div", "itruediv": "div", "rtruediv": "div", "matmul": "matmul",
    "rmatmul": "matmul", "getitem": "index", "setitem": "index_put",
}  # fmt: skip


def _op_name(func: Any) -> str:
    name = getattr(func, "__name__", None) or str(func)
    if name == "__get__":  # property access such as ``x.T`` or ``x.mT``
        name = getattr(getattr(func, "__self__", None), "__name__", "attr")
    name = name.strip("_")
    return _OP_ALIASES.get(name, name)


def _tensors(obj: Any) -> Iterator[torch.Tensor]:
    """Yield every tensor inside nested tuples, lists and dicts."""
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (tuple, list)):
        for o in obj:
            yield from _tensors(o)
    elif isinstance(obj, dict):
        for o in obj.values():
            yield from _tensors(o)


def _module_role(module: nn.Module) -> str:
    if isinstance(module, PASSTHROUGH_TYPES):
        return "passthrough"
    if isinstance(module, ACTIVATION_TYPES):
        return "activation"
    if isinstance(module, NORM_TYPES):
        return "norm"
    if isinstance(module, RESHAPE_TYPES):
        return "reshape"
    return "layer"


@dataclass
class _Frame:
    module: nn.Module
    inputs: list[int]
    first_node: int
    atomic: bool
    inside_atomic: bool


class _OpRecorder(TorchFunctionMode):
    def __init__(self, tracer: _Tracer):
        super().__init__()
        self.tracer = tracer

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        self.tracer.on_op(func, args, kwargs, out)
        return out


class _Tracer:
    def __init__(self, model: nn.Module, atomic: tuple[type[nn.Module], ...]):
        self.model = model
        self.atomic = ATOMIC_TYPES + atomic
        self.paths = {m: name for name, m in model.named_modules()}
        self.nodes: list[Node] = []
        self.producer: dict[int, int] = {}  # id(tensor) -> node id
        self.keep: list[torch.Tensor] = []  # keeps tracked tensors alive so ids stay unique
        self.stack: list[_Frame] = []
        self.atomic_depth = 0
        self.calls: Counter[nn.Module] = Counter()
        self.busy = False

    # -- graph building -------------------------------------------------------------

    def track(self, t: torch.Tensor, node_id: int) -> None:
        self.producer[id(t)] = node_id
        self.keep.append(t)

    def input_ids(self, *objs: Any) -> list[int]:
        ids = [self.producer[id(t)] for o in objs for t in _tensors(o) if id(t) in self.producer]
        return list(dict.fromkeys(ids))

    def add(self, outs: list[torch.Tensor], **fields: Any) -> Node:
        self.busy = True
        try:
            summary = summarize(outs[0], image=fields.get("role") == "input")
        finally:
            self.busy = False
        node = Node(id=len(self.nodes), shape=tuple(outs[0].shape), summary=summary, **fields)
        self.nodes.append(node)
        for t in outs:
            self.track(t, node.id)
        return node

    def scope(self) -> tuple[str, ...]:
        for frame in reversed(self.stack):
            path = self.paths.get(frame.module, "")
            return tuple(path.split(".")) if path else ()
        return ()

    # -- hooks ----------------------------------------------------------------------

    def pre_hook(self, module: nn.Module, args: Any, kwargs: Any) -> None:
        atomic = isinstance(module, self.atomic) or not any(True for _ in module.children())
        frame = _Frame(
            module=module,
            inputs=self.input_ids(args, kwargs),
            first_node=len(self.nodes),
            atomic=atomic,
            inside_atomic=self.atomic_depth > 0,
        )
        self.stack.append(frame)
        self.atomic_depth += atomic

    def post_hook(self, module: nn.Module, args: Any, kwargs: Any, output: Any) -> None:
        frame = self.stack.pop()
        self.atomic_depth -= frame.atomic
        if frame.inside_atomic or not frame.inputs:
            return
        outs = list(_tensors(output))
        if not outs:
            return
        # A container whose forward recorded nothing (e.g. a fused fast path) is drawn whole.
        if frame.atomic or len(self.nodes) == frame.first_node:
            self.calls[module] += 1
            path = self.paths.get(module, "") or type(module).__name__
            call = self.calls[module]
            name = path if call == 1 else f"{path}:{call}"
            scope = tuple(name.split("."))
            self.add(
                outs,
                kind="module",
                name=name,
                type=type(module).__name__,
                role=_module_role(module),
                scope=scope,
                inputs=frame.inputs,
                info=f"{type(module).__name__}({module.extra_repr()})",
                params=sum(p.numel() for p in module.parameters()),
                module=module,
            )

    def on_op(self, func: Any, args: Any, kwargs: Any, out: Any) -> None:
        if self.busy or self.atomic_depth:
            return
        outs = list(_tensors(out))
        if not outs:
            return
        tracked = [t for t in _tensors((args, kwargs)) if id(t) in self.producer]
        if not tracked:
            return
        inputs = list(dict.fromkeys(self.producer[id(t)] for t in tracked))
        name = _op_name(func)
        same_shape = len(inputs) == 1 and outs[0].shape == tracked[0].shape
        if same_shape and name in SILENT_OPS:
            for t in outs:
                self.track(t, inputs[0])
            return
        if len(inputs) > 1:
            role = "merge"
        elif name in ACTIVATION_OPS:
            role = "activation"
        elif name in NORM_OPS:
            role = "norm"
        else:
            role = "elementwise" if same_shape else "reshape"
        self.add(
            outs,
            kind="op",
            name=name,
            type=name,
            role=role,
            scope=self.scope() + (name,),
            inputs=inputs,
        )


@contextmanager
def _eval_mode(model: nn.Module) -> Iterator[None]:
    flags = [(m, m.training) for m in model.modules()]
    model.eval()
    try:
        yield
    finally:
        for m, was_training in flags:
            m.training = was_training


def trace(
    model: nn.Module,
    inputs: torch.Tensor | tuple[Any, ...],
    *,
    simplify_graph: bool = True,
    atomic: tuple[type[nn.Module], ...] = (),
) -> Graph:
    """Run one forward pass and record it as a dataflow graph.

    The model runs in eval mode under ``torch.no_grad()``. Its training flags are
    restored and every hook is removed afterwards, even on error.

    Args:
        model: Any ``nn.Module``.
        inputs: A batch tensor, or a tuple of positional arguments for ``model``.
        simplify_graph: Drop dropout/identity, fold activations and norms into the layer
            before them, and merge reshape chains (see ``graph.simplify``).
        atomic: Extra module types to draw as a single node instead of their children.

    Returns:
        The graph, with nodes in execution order.

    Example:
        >>> graph = trace(model, torch.randn(4, 3, 32, 32))
        >>> print(graph)
    """
    args = inputs if isinstance(inputs, tuple) else (inputs,)
    tracer = _Tracer(model, atomic)
    for i, t in enumerate(_tensors(args)):
        name = "input" if i == 0 else f"input.{i}"
        tracer.add(
            [t], kind="input", name=name, type="Input", role="input", scope=(name,), inputs=[]
        )

    handles = []
    try:
        for m in model.modules():
            handles.append(m.register_forward_pre_hook(tracer.pre_hook, with_kwargs=True))
            handles.append(m.register_forward_hook(tracer.post_hook, with_kwargs=True))
        with _eval_mode(model), torch.no_grad(), _OpRecorder(tracer):
            output = model(*args)
    finally:
        for h in handles:
            h.remove()

    for t in _tensors(output):
        if id(t) in tracer.producer:
            tracer.nodes[tracer.producer[id(t)]].is_output = True
    graph = Graph(tracer.nodes)
    return simplify(graph) if simplify_graph else graph
