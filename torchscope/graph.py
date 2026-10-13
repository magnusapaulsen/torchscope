"""Dataflow graph of a traced forward pass, and its simplification."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import nn

from torchscope.summary import Summary

# Roles that fold into the node before them when that node has a single consumer.
FOLDABLE_ROLES = ("activation", "norm", "elementwise")


@dataclass(eq=False)
class Node:
    """One module call or tensor op in the forward pass.

    Attributes:
        id: Index in ``Graph.nodes``, which is in execution (topological) order.
        kind: ``"input"``, ``"module"`` or ``"op"``.
        name: Module path (``"layer1.0.conv1"``) or op name (``"add"``).
        type: Module class name (``"Conv2d"``) or op name.
        role: ``"input"``, ``"layer"``, ``"activation"``, ``"norm"``, ``"passthrough"``,
            ``"reshape"``, ``"elementwise"`` or ``"merge"``. Drives simplification and drawing.
        scope: Module path the node was created in, ending with its own name. Used to
            collapse nested modules into groups.
        inputs: Ids of the nodes whose outputs this node consumes.
        shape: Output shape, batch dim included.
        info: Module ``repr`` line, e.g. ``"Conv2d(3, 8, kernel_size=(3, 3))"``.
        params: Parameter count of the module (and anything folded into it).
        tags: Types of the nodes folded into this one, e.g. ``["BatchNorm2d", "ReLU"]``.
        is_output: Whether this node produces a model output.
        summary: Downsampled activations of the output, for the viewer.
        module: The module itself, for module nodes.
    """

    id: int
    kind: str
    name: str
    type: str
    role: str
    scope: tuple[str, ...]
    inputs: list[int]
    shape: tuple[int, ...]
    info: str = ""
    params: int = 0
    tags: list[str] = field(default_factory=list)
    is_output: bool = False
    summary: Summary | None = field(default=None, repr=False)
    module: nn.Module | None = field(default=None, repr=False)

    @property
    def label(self) -> str:
        """Short display name: the last scope component, or the op name."""
        return self.scope[-1] if self.scope else self.name


@dataclass
class Graph:
    """Nodes in execution order. Edges are stored as ``Node.inputs``."""

    nodes: list[Node]

    def consumers(self) -> dict[int, list[int]]:
        """Map each node id to the ids of the nodes that consume its output."""
        out: dict[int, list[int]] = {n.id: [] for n in self.nodes}
        for n in self.nodes:
            for i in n.inputs:
                out[i].append(n.id)
        return out

    def edges(self) -> list[tuple[int, int]]:
        """All ``(source, target)`` pairs."""
        return [(i, n.id) for n in self.nodes for i in n.inputs]

    def __str__(self) -> str:
        """One line per node: id, name, type, tags, output shape and inputs."""
        lines = []
        for n in self.nodes:
            tags = f" +{'+'.join(n.tags)}" if n.tags else ""
            out = " [out]" if n.is_output else ""
            lines.append(f"{n.id:3d} {n.name} ({n.type}{tags}) {list(n.shape)} <- {n.inputs}{out}")
        return "\n".join(lines)


def simplify(graph: Graph) -> Graph:
    """Return a cleaner copy of ``graph`` for display.

    Drops pass-through nodes (dropout, identity), folds activations, norms and
    shape-preserving unary ops into the node before them when that node has a single
    consumer, and merges chains of reshape-like ops. Node ids are renumbered.

    Args:
        graph: A raw graph from ``trace(..., simplify=False)``.

    Returns:
        A new graph. Nodes are shared with ``graph`` only through their summaries/modules.
    """
    nodes = {
        n.id: Node(**{**n.__dict__, "inputs": list(n.inputs), "tags": list(n.tags)})
        for n in graph.nodes
    }

    def consumers_of(nid: int) -> list[int]:
        return [m.id for m in nodes.values() if nid in m.inputs]

    def bypass(n: Node, into: Node) -> None:
        for m in nodes.values():
            m.inputs = [into.id if i == n.id else i for i in m.inputs]
            m.inputs = list(dict.fromkeys(m.inputs))
        into.is_output = into.is_output or n.is_output
        del nodes[n.id]

    for n in list(nodes.values()):
        if n.role == "passthrough" and len(n.inputs) == 1:
            bypass(n, nodes[n.inputs[0]])

    for n in list(nodes.values()):
        if len(n.inputs) != 1:
            continue
        prev = nodes[n.inputs[0]]
        if prev.role == "input" or consumers_of(prev.id) != [n.id]:
            continue
        foldable = n.role in FOLDABLE_ROLES and prev.shape[1:] == n.shape[1:]
        reshape_chain = n.role == "reshape" and prev.role == "reshape"
        if not (foldable or reshape_chain):
            continue
        if reshape_chain:
            prev.name = prev.type = f"{prev.type} → {n.type}"
            prev.scope = prev.scope[:-1] + (prev.type,)
        else:
            prev.tags.append(n.type)
        prev.shape, prev.summary = n.shape, n.summary
        prev.params += n.params
        bypass(n, prev)

    remap = {old: new for new, old in enumerate(nodes)}
    out = []
    for n in nodes.values():
        n.id = remap[n.id]
        n.inputs = [remap[i] for i in n.inputs]
        out.append(n)
    return Graph(out)
