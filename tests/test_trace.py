import re

import pytest
import torch
from conftest import MODELS, ResNetish, TinyCNN, TinyViT, assert_unchanged, encoder, mlp, snapshot
from torch import nn

from torchscope import trace
from torchscope.trace import ATOMIC_TYPES


def types(graph):
    return [n.type for n in graph.nodes]


def test_mlp_folds_activations():
    g = trace(mlp(), torch.randn(2, 1, 8, 8))
    assert types(g) == ["Input", "Flatten", "Linear", "Linear", "Linear"]
    assert [n.tags for n in g.nodes[2:]] == [["ReLU"], ["ReLU"], []]
    assert [n.inputs for n in g.nodes] == [[], [0], [1], [2], [3]]
    assert g.nodes[-1].is_output and g.nodes[-1].shape == (2, 4)


def test_cnn_folds_norms_and_drops_dropout():
    g = trace(TinyCNN(), torch.randn(2, 1, 8, 8))
    conv1 = next(n for n in g.nodes if n.name == "conv1")
    assert conv1.tags == ["BatchNorm2d", "relu"]
    assert "Dropout" not in types(g)
    flatten = next(n for n in g.nodes if n.type == "flatten")
    assert flatten.role == "reshape" and flatten.shape == (2, 32)


def test_residual_add_is_a_merge_node():
    g = trace(ResNetish(), torch.randn(2, 3, 8, 8))
    adds = [n for n in g.nodes if n.type == "add"]
    assert len(adds) == 2
    for add in adds:
        assert add.role == "merge" and len(add.inputs) == 2
        assert add.tags == ["ReLU"]  # the second call of the shared ReLU folds into the add
        assert add.scope[:2] in (("layer1", "0"), ("layer1", "1"))


def test_shared_module_calls_are_separate_nodes():
    g = trace(ResNetish(), torch.randn(2, 3, 8, 8), simplify_graph=False)
    names = [n.name for n in g.nodes if n.type == "ReLU"]
    assert names == ["layer1.0.relu", "layer1.0.relu:2", "layer1.1.relu", "layer1.1.relu:2"]


def test_vit_attention_is_atomic():
    g = trace(TinyViT(), torch.randn(2, 3, 16, 16))
    attn = [n for n in g.nodes if n.type == "MultiheadAttention"]
    assert len(attn) == 2 and all(n.shape == (2, 17, 8) for n in attn)
    assert nn.MultiheadAttention in ATOMIC_TYPES
    # Nothing inside the attention module leaks into the graph.
    assert not any(n.name.startswith("blocks.0.attn.") for n in g.nodes)
    stem = g.nodes[2]
    assert stem.role == "reshape" and stem.shape == (2, 17, 8)
    assert g.nodes[-1].type == "Linear" and g.nodes[-2].type == "index"


def test_transformer_encoder_layers_are_traced():
    g = trace(encoder(), torch.randn(2, 5, 8))
    assert types(g).count("MultiheadAttention") == 2
    assert g.nodes[-1].is_output and g.nodes[-1].shape == (2, 5, 8)


def test_container_without_recorded_ops_becomes_one_node():
    class Fused(nn.Module):
        def __init__(self):
            super().__init__()
            self.unused = nn.Linear(4, 4)

        def forward(self, x):
            return x.clone()

    g = trace(nn.Sequential(nn.Linear(4, 4), Fused()), torch.randn(2, 4))
    assert types(g) == ["Input", "Linear", "Fused"]
    assert g.nodes[-1].params == 20


def test_custom_atomic_types():
    from conftest import Block

    g = trace(ResNetish(), torch.randn(2, 3, 8, 8), atomic=(Block,))
    assert types(g) == ["Input", "Conv2d", "Block", "Block", "mean", "Linear"]


@pytest.mark.parametrize("name", list(MODELS))
def test_every_model_is_a_dag_in_execution_order(name):
    factory, shape = MODELS[name]
    g = trace(factory(), torch.randn(shape))
    for n in g.nodes:
        assert n.id == g.nodes.index(n)
        assert all(i < n.id for i in n.inputs)
    assert any(n.is_output for n in g.nodes)


def test_model_is_left_unchanged(case):
    _, model, x = case
    model.train()
    for m in list(model.modules())[1::2]:
        m.eval()  # mixed training flags must survive
    before = snapshot(model)
    trace(model, x)
    assert_unchanged(model, before)


def test_hooks_and_mode_removed_on_error():
    class Broken(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)

        def forward(self, x):
            return self.lin(x) @ torch.randn(3, 3)  # shape error

    model = Broken()
    before = snapshot(model)
    with pytest.raises(RuntimeError):
        trace(model, torch.randn(2, 4))
    assert_unchanged(model, before)


def test_multiple_inputs():
    class Two(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(4, 3)
            self.b = nn.Linear(5, 3)

        def forward(self, x, y):
            return self.a(x) * self.b(y)

    g = trace(Two(), (torch.randn(2, 4), torch.randn(2, 5)))
    assert [n.name for n in g.nodes] == ["input", "input.1", "a", "b", "mul"]
    assert g.nodes[-1].inputs == [2, 3]


def available_devices():
    devices = []
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.mark.skipif(not available_devices(), reason="no accelerator")
@pytest.mark.parametrize("device", available_devices())
def test_trace_runs_on_the_inputs_device(device):
    import torchscope

    model = TinyCNN().to(device)
    viz = torchscope.visualize(model, torch.randn(2, 1, 8, 8, device=device))
    assert viz.view == "slabs"
    assert all(p.device.type == device for p in model.parameters())


def test_no_hardcoded_devices():
    from pathlib import Path

    import torchscope

    for path in Path(torchscope.__file__).parent.glob("*.py"):
        text = path.read_text()
        assert ".cuda(" not in text, path.name
        assert not re.search(r"device\(?\s*=?\s*[\"']cuda|\.to\([\"']cuda", text), path.name
