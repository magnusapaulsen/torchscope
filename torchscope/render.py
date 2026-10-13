"""Inline the viewer assets and the payload into one self-contained HTML page."""

from __future__ import annotations

import html
import json
from collections.abc import Sequence
from importlib import resources
from pathlib import Path
from typing import Any

_JS_FILES = ("common.js", "learn.js", "neurons.js", "slabs.js", "main.js")


def _asset(name: str) -> str:
    return resources.files("torchscope").joinpath("viewer", name).read_text(encoding="utf-8")


def render_html(payload: dict[str, Any]) -> str:
    """Return the viewer page for ``payload`` as an HTML string."""
    data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    script = "\n".join(_asset(name) for name in _JS_FILES)
    return (
        _asset("template.html")
        .replace("{{TITLE}}", html.escape(payload.get("title", "TorchScope")))
        .replace("{{CSS}}", _asset("style.css"))
        .replace("{{SCRIPT}}", script)
        .replace("{{DATA}}", data)
    )


class Visualization:
    """An interactive view of a traced model. Save it, or display it in Jupyter.

    Attributes:
        payload: The JSON-serializable data the viewer renders.
        graph: The simplified graph it was built from.
    """

    def __init__(self, payload: dict[str, Any], graph: Any):
        """Wrap a payload from ``export.build_payload`` and its graph."""
        self.payload = payload
        self.graph = graph

    @property
    def view(self) -> str:
        """``"neurons"`` or ``"slabs"``: the view the page opens with."""
        return self.payload["view"]

    def to_html(self) -> str:
        """Return the page as a self-contained HTML string."""
        return render_html(self.payload)

    def save(self, path: str | Path) -> Path:
        """Write the page to ``path`` and return the path.

        Args:
            path: Destination file, usually ending in ``.html``.

        Returns:
            The path that was written.
        """
        path = Path(path)
        path.write_text(self.to_html(), encoding="utf-8")
        return path

    def to_svg(
        self,
        *,
        samples: Sequence[int] | None = None,
        width: int = 960,
        height: int = 540,
        title: bool = True,
    ) -> str:
        """Return an animated SVG of the view, for READMEs.

        The animation is pure CSS, so it plays where scripts are not allowed, such as a
        GitHub README. It cycles through ``samples`` one forward pass at a time.

        Args:
            samples: Indices of the samples to animate. Defaults to the first three.
            width: Width in pixels.
            height: Height in pixels.
            title: Draw the title and parameter count in the top-left corner.

        Returns:
            The SVG document as a string.
        """
        from torchscope.svg import render_svg

        return render_svg(self.payload, samples=samples, width=width, height=height, title=title)

    def save_svg(self, path: str | Path, **kwargs: Any) -> Path:
        """Write ``to_svg(**kwargs)`` to ``path`` and return the path.

        Example:
            >>> viz.save_svg("docs/network.svg")  # then in README.md: ![](docs/network.svg)
        """
        path = Path(path)
        path.write_text(self.to_svg(**kwargs), encoding="utf-8")
        return path

    def _repr_html_(self) -> str:
        srcdoc = html.escape(self.to_html(), quote=True)
        return (
            f'<iframe srcdoc="{srcdoc}" style="width:100%;height:640px;border:0;'
            'border-radius:8px" sandbox="allow-scripts"></iframe>'
        )

    def __repr__(self) -> str:
        """Short description: model name, view and node count."""
        return (
            f"<Visualization {self.payload['model']['name']} view={self.view!r} "
            f"nodes={len(self.graph.nodes)}>"
        )
