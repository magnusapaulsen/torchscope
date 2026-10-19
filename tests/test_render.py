import json
import re
import shutil
import subprocess

import pytest
import torch
from conftest import TinyCNN, mlp

import torchscope
from torchscope.render import _JS_FILES, _asset


def embedded_payload(html):
    match = re.search(r'<script id="payload" type="application/json">(.*?)</script>', html, re.S)
    return json.loads(match.group(1))


def test_html_is_self_contained(tmp_path):
    viz = torchscope.visualize(TinyCNN(), torch.randn(2, 1, 8, 8), title="My CNN")
    path = viz.save(tmp_path / "net.html")
    html = path.read_text()
    assert "{{" not in html
    assert "<title>My CNN</title>" in html
    assert embedded_payload(html) == json.loads(json.dumps(viz.payload))
    assert "three@0.160.0" in html


def test_script_injection_is_escaped():
    viz = torchscope.visualize(
        mlp(), torch.randn(2, 1, 8, 8), labels=["</script><b>x", "b", "c", "d"]
    )
    html = viz.to_html()
    assert html.count("</script>") == html.count("<script")
    assert embedded_payload(html)["neurons"]["output"]["labels"][0] == "</script><b>x"


def test_jupyter_repr_is_an_iframe():
    viz = torchscope.visualize(mlp(), torch.randn(2, 1, 8, 8))
    out = viz._repr_html_()
    assert out.startswith("<iframe srcdoc=") and "&lt;script" in out


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_viewer_script_parses(tmp_path):
    script = tmp_path / "viewer.mjs"
    script.write_text("\n".join(_asset(name) for name in _JS_FILES))
    subprocess.run(["node", "--check", str(script)], check=True)


def test_neuron_view_needs_no_network():
    html = torchscope.visualize(mlp(), torch.rand(3, 1, 8, 8)).to_html()
    assert not re.search(r"<(script|link|img)\b[^>]*\b(src|href)=[\"']https?:", html)
    # three.js is only reachable through the import map, and only imported dynamically.
    scripts = "\n".join(_asset(name) for name in _JS_FILES)
    assert not re.search(r"^\s*import\b", scripts, re.M)
    assert scripts.count('import("three') == scripts.count("await import(")
