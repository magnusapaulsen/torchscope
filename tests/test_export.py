import base64
import json

import pytest
import torch
from conftest import MODELS, TinyCNN, TinyViT, mlp
from torch import nn

import torchscope
from torchscope import trace
from torchscope.export import is_mlp


def decoded(b64):
    return base64.b64decode(b64)


@pytest.mark.parametrize(
    "name, expected",
    [("mlp", True), ("cnn", False), ("resnet", False), ("vit", False), ("encoder", False)],
)
def test_view_selection(name, expected):
    factory, shape = MODELS[name]
    graph = trace(factory(), torch.randn(shape))
    assert is_mlp(graph) is expected
    assert torchscope.visualize(factory(), torch.randn(shape)).view == (
        "neurons" if expected else "slabs"
    )


def test_neuron_payload():
    viz = torchscope.visualize(mlp(), torch.rand(3, 1, 8, 8), max_neurons=10, labels=list("abcd"))
    n = viz.payload["neurons"]
    sizes = [layer["size"] for layer in n["layers"]]
    assert sizes == [64, 12, 8, 4]
    shown = [len(layer["shown"]) for layer in n["layers"]]
    assert shown == [10, 10, 8, 4]
    for prev, layer in zip(n["layers"], n["layers"][1:], strict=False):
        assert len(layer["weights"]) == len(layer["shown"])
        assert all(len(row) == len(prev["shown"]) for row in layer["weights"])
        assert len(layer["values"]) == 3 and len(layer["values"][0]) == len(layer["shown"])
    image = n["image"]
    assert (image["c"], image["h"], image["w"]) == (1, 8, 8)
    assert all(len(decoded(d)) == 64 for d in image["data"])
    out = n["output"]
    assert out["labels"] == list("abcd")
    assert all(abs(sum(p) - 1) < 1e-3 for p in out["probs"])


def test_shown_neurons_are_the_most_active():
    model = nn.Sequential(nn.Linear(6, 3))
    x = torch.zeros(2, 6)
    x[:, 4] = 5.0
    x[:, 1] = 3.0
    n = torchscope.visualize(model, x, max_neurons=2).payload["neurons"]
    assert n["layers"][0]["shown"] == [1, 4]


def test_flat_input_drawn_as_image_on_request():
    model = nn.Sequential(nn.Linear(16, 3))
    assert torchscope.visualize(model, torch.rand(2, 16)).payload["neurons"]["image"] is None
    n = torchscope.visualize(model, torch.rand(2, 16), image_shape=(4, 4)).payload["neurons"]
    assert (n["image"]["h"], n["image"]["w"]) == (4, 4)


def test_neurons_view_rejected_for_cnn():
    with pytest.raises(ValueError, match="neuron view"):
        torchscope.visualize(TinyCNN(), torch.randn(2, 1, 8, 8), view="neurons")


def test_graph_payload_faces_and_predictions():
    payload = torchscope.visualize(
        TinyViT(), torch.randn(3, 3, 16, 16), labels=list("vwxyz")
    ).payload
    nodes = payload["graph"]["nodes"]
    assert payload["samples"] == 3
    for node in nodes:
        for face in node["faces"].values():
            assert len(face["data"]) == 3
            assert all(len(decoded(d)) == face["h"] * face["w"] for d in face["data"])
    image = nodes[0]["image"]
    assert all(len(decoded(d)) == image["h"] * image["w"] * 3 for d in image["data"])
    out = [n for n in nodes if n["output"]]
    assert len(out) == 1 and len(out[0]["top"]) == 3
    label, p = out[0]["top"][0][0]
    assert label in "vwxyz" and 0 < p <= 1
    json.dumps(payload)  # serializable


def test_max_samples():
    viz = torchscope.visualize(TinyCNN(), torch.randn(20, 1, 8, 8), max_samples=4)
    assert viz.payload["samples"] == 4
    assert viz.graph.nodes[0].shape[0] == 4
