"""Payload fields behind the learning features: maths, captions, feature maps, convolution."""

import torch
import torch.nn.functional as F
from conftest import ResNetish, TinyCNN, TinyViT, mlp
from torch import nn

import torchscope
from torchscope import trace
from torchscope.explain import describe
from torchscope.summary import CHANNELS_MAX


def test_neuron_maths_matches_the_layer():
    model = mlp()
    x = torch.rand(3, 1, 8, 8)
    layers = torchscope.visualize(model, x, max_neurons=6).payload["neurons"]["layers"]
    first = model[1]
    z = x.flatten(1) @ first.weight.T + first.bias
    hidden = layers[1]
    assert torch.allclose(torch.tensor(hidden["z"]), z[:, hidden["shown"]], atol=1e-4)
    assert hidden["act"] == "ReLU" and hidden["exact"]
    assert torch.allclose(torch.tensor(hidden["values"]), z.relu()[:, hidden["shown"]], atol=1e-4)
    # The listed terms are the largest |w * a| for that neuron, largest first.
    terms = hidden["terms"][0][0]
    products = (x[0].flatten() * first.weight[hidden["shown"][0]]).abs()
    assert [i for i, _, _ in terms] == products.topk(len(terms)).indices.tolist()
    assert layers[-1]["act"] is None


def test_neuron_captions_use_real_sizes():
    layers = torchscope.visualize(mlp(), torch.rand(2, 1, 8, 8)).payload["neurons"]["layers"]
    assert "64 pixel values" in layers[0]["explain"]
    assert "12 neurons" in layers[1]["explain"] and "64 neurons before" in layers[1]["explain"]
    assert "ReLU" in layers[1]["explain"]
    assert "prediction" in layers[-1]["explain"]


def test_layer_captions():
    g = trace(TinyCNN(), torch.rand(2, 1, 8, 8))
    by_type = {n.type: describe(n, g) for n in g.nodes}
    assert "8 filters, each 3×3 (×4 input channels)" in by_type["Conv2d"]
    assert "2×2 window" in by_type["MaxPool2d"]
    assert "unrolls the 8×2×2 block into one list of 32 numbers" in by_type["flatten"]
    assert "8×8 grayscale image" in describe(g.nodes[0], g)
    assert "BatchNorm" in describe(g.nodes[1], g) and "ReLU" in describe(g.nodes[1], g)
    r = trace(ResNetish(), torch.rand(2, 3, 8, 8))
    add = next(n for n in r.nodes if n.type == "add")
    assert "skip" in describe(add, r)
    v = trace(TinyViT(), torch.rand(2, 3, 16, 16))
    attn = next(n for n in v.nodes if n.type == "MultiheadAttention")
    assert "attention" in describe(attn, v).lower()


def test_every_node_has_a_caption():
    nodes = torchscope.visualize(TinyViT(), torch.rand(2, 3, 16, 16)).payload["graph"]["nodes"]
    assert all(n["explain"] for n in nodes)


def test_feature_maps_per_channel():
    nodes = torchscope.visualize(TinyCNN(), torch.rand(2, 1, 8, 8)).payload["graph"]["nodes"]
    conv1 = next(n for n in nodes if n["name"] == "conv1")
    ch = conv1["channels"]
    assert (ch["c"], ch["h"], ch["w"], ch["total"]) == (4, 8, 8, 4)
    assert len(ch["data"]) == 2
    # Token and vector outputs have no channel maps.
    assert "channels" not in next(n for n in nodes if n["type"] == "Linear")
    big = torchscope.visualize(nn.Conv2d(3, 64, 1), torch.rand(1, 3, 8, 8)).payload["graph"][
        "nodes"
    ]
    assert big[1]["channels"]["c"] == CHANNELS_MAX and big[1]["channels"]["total"] == 64


def test_convolution_payload_reproduces_the_layer():
    model = nn.Sequential(nn.Conv2d(2, 3, 3, padding=1, stride=2), nn.ReLU())
    x = torch.rand(2, 2, 7, 7)
    conv = torchscope.visualize(model, x).payload["graph"]["nodes"][1]["conv"]
    inp = torch.tensor(conv["in"]["data"]).reshape(
        2, conv["in"]["c"], conv["in"]["h"], conv["in"]["w"]
    )
    out = F.conv2d(
        inp,
        torch.tensor(conv["weight"]),
        torch.tensor(conv["bias"]),
        conv["stride"],
        conv["padding"],
    )
    assert torch.allclose(out.relu(), model(x), atol=1e-3)
    assert conv["act"] == "ReLU" and conv["exact"]


def test_convolution_payload_skipped_when_not_reproducible():
    nodes = torchscope.visualize(TinyCNN(), torch.rand(2, 1, 8, 8)).payload["graph"]["nodes"]
    conv1 = next(n for n in nodes if n["name"] == "conv1")
    assert conv1["conv"]["exact"] is False  # BatchNorm is folded in, so z is not the output
    grouped = nn.Conv2d(4, 4, 3, groups=2)
    assert (
        "conv"
        not in torchscope.visualize(grouped, torch.rand(1, 4, 6, 6)).payload["graph"]["nodes"][1]
    )
