import base64
import re
import xml.etree.ElementTree as ET

import pytest
import torch
from conftest import TinyCNN, mlp

import torchscope
from torchscope.svg import _png

SVG = "{http://www.w3.org/2000/svg}"


def parse(svg):
    return ET.fromstring(svg)


def duration(svg):
    return float(re.search(r"animation:a0 ([\d.]+)s", svg).group(1))


def test_mlp_svg_structure():
    viz = torchscope.visualize(mlp(), torch.rand(4, 1, 8, 8), max_neurons=10)
    svg = viz.to_svg()
    root = parse(svg)
    assert root.tag == f"{SVG}svg" and root.get("viewBox") == "0 0 960 540"
    shown = sum(len(layer["shown"]) for layer in viz.payload["neurons"]["layers"])
    shade = root.find(f".//{SVG}g[@fill='url(#shade)']")
    assert len(shade.findall(f"{SVG}circle")) == shown
    images = root.findall(f".//{SVG}image")
    assert len(images) == 3  # one input image per animated sample (default: first three)
    assert all(img.get("href").startswith("data:image/png;base64,") for img in images)
    assert "<script" not in svg and "@keyframes" in svg


def test_samples_cycle_and_validation():
    viz = torchscope.visualize(mlp(), torch.rand(4, 1, 8, 8))
    one = duration(viz.to_svg(samples=[2]))
    assert duration(viz.to_svg(samples=[0, 1, 3])) == pytest.approx(3 * one, rel=1e-3)
    with pytest.raises(ValueError, match="samples"):
        viz.to_svg(samples=[4])


def test_cnn_svg_has_textured_faces_and_prediction():
    viz = torchscope.visualize(TinyCNN(), torch.rand(2, 1, 8, 8), labels=list("abcde"))
    svg = viz.to_svg(samples=[0, 1], title=False)
    root = parse(svg)
    images = root.findall(f".//{SVG}image")
    assert images and all(img.get("transform", "").startswith("matrix(") for img in images)
    assert "prediction" in svg
    assert "<script" not in svg


def test_save_svg(tmp_path):
    viz = torchscope.visualize(mlp(), torch.rand(2, 1, 8, 8))
    path = viz.save_svg(tmp_path / "net.svg", width=640, height=360)
    assert parse(path.read_text()).get("width") == "640"


def test_png_encoder_round_trip():
    pil = pytest.importorskip("PIL.Image")
    import io

    pixels = bytes(range(0, 240, 20)) * 3  # 4x3 RGB
    uri = _png(pixels, 4, 3, 3)
    img = pil.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert img.size == (4, 3) and img.mode == "RGB"
    assert img.tobytes() == pixels
