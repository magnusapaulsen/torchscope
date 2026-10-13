"""Animated SVG export. The animation is pure CSS, so it plays inside a GitHub README."""

from __future__ import annotations

import base64
import struct
import zlib
from collections.abc import Sequence
from typing import Any
from xml.sax.saxutils import escape

BLUE = (35, 190, 240)
RED = (250, 80, 40)
ACCENT = (88, 196, 221)
BG = "#000"
TEXT = "#eceef1"
MUTED = "#8d939c"
FONT = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

# Neuron-view timeline, per sample (seconds).
T_IN, T_EDGE, T_FIRE, T_HOLD, T_GAP = 0.8, 1.1, 0.35, 1.8, 0.5


def _f(x: float) -> str:
    """Compact number formatting for coordinates."""
    s = f"{x:.1f}"
    return s[:-2] if s.endswith(".0") else s


def _rgb(c: Sequence[float]) -> str:
    return "#" + "".join(f"{round(min(255, max(0, v))):02x}" for v in c)


def _png(pixels: bytes, w: int, h: int, channels: int) -> str:
    """Encode 8-bit grayscale or RGB pixels as a PNG data URI."""
    stride = w * channels
    raw = b"".join(b"\x00" + pixels[y * stride : (y + 1) * stride] for y in range(h))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    color = 2 if channels == 3 else 0
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, color, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


class _Timeline:
    """CSS keyframe animations that all loop over the same total duration."""

    def __init__(self, total: float):
        self.total = total
        self.rules: list[str] = []

    def add(self, frames: list[tuple[float, str]]) -> str:
        """Add an animation from ``(time, declarations)`` frames; returns its class name."""
        name = f"a{len(self.rules)}"
        frames = [(0.0, frames[0][1]), *frames, (self.total, frames[-1][1])]
        steps, last = [], -1.0
        for t, css in frames:
            pct = max(last, min(100.0, 100 * t / self.total))
            steps.append(f"{pct:.3f}%{{{css}}}")
            last = pct
        self.rules.append(
            f"@keyframes {name}{{{''.join(steps)}}}"
            f".{name}{{animation:{name} {self.total:.2f}s linear infinite}}"
        )
        return name

    def fade(self, t0: float, t1: float, t2: float, t3: float) -> str:
        """Opacity 0 until ``t0``, 1 from ``t1`` to ``t2``, 0 again after ``t3``."""
        return self.add(
            [(t0, "opacity:0"), (t1, "opacity:1"), (t2, "opacity:1"), (t3, "opacity:0")]
        )

    def css(self) -> str:
        """All rules, ready for a ``<style>`` element."""
        return "".join(self.rules)


def _brightness_scale(values: list[list[float]]) -> float:
    """90th percentile of the non-zero magnitudes, like the HTML viewer."""
    mags = sorted(abs(v) for row in values for v in row if abs(v) > 1e-9)
    return mags[min(len(mags) - 1, int(len(mags) * 0.9))] if mags else 1.0


def neuron_svg(
    payload: dict[str, Any],
    samples: Sequence[int],
    width: int,
    height: int,
    title: bool,
) -> str:
    """Render the neuron view as an animated SVG that cycles through ``samples``."""
    n = payload["neurons"]
    layers = n["layers"]
    out = n["output"]
    image = n["image"]
    L = len(layers)
    starts = [0.0] + [T_IN + (k - 1) * (T_EDGE + T_FIRE) for k in range(1, L)]
    period = T_IN + (L - 1) * (T_EDGE + T_FIRE) + T_HOLD + T_GAP
    tl = _Timeline(period * len(samples))

    # Layout, mirroring the HTML neuron view.
    top = 96 if title else 56
    bottom = 40
    avail = height - top - bottom
    max_shown = max(len(layer["shown"]) for layer in layers)
    r = min(13.0, max(2.5, avail / (max_shown * 2.6)))
    img_box = None
    if image:
        size = min(avail * 0.62, width * 0.16)
        aspect = image["w"] / image["h"]
        iw, ih = (size, size / aspect) if aspect >= 1 else (size * aspect, size)
        img_box = (28.0, top + avail / 2 - ih / 2, iw, ih)
    left = (img_box[0] + img_box[2] + max(28, width * 0.05) if img_box else 40) + r
    right = width - (110 if out else 30) - r
    xs = [left if L == 1 else left + (right - left) * k / (L - 1) for k in range(L)]
    cols = []
    for k, layer in enumerate(layers):
        count = len(layer["shown"])
        gap = min(r * 2.7, avail / max(1, count))
        y0 = top + avail / 2 - gap * (count - 1) / 2
        cols.append([(xs[k], y0 + j * gap) for j in range(count)])

    wmax = [
        max((abs(w) for row in layer.get("weights") or [] for w in row), default=1) or 1
        for layer in layers
    ]
    edges = [[]] + [
        sorted(
            ((i, j, w) for j, row in enumerate(layer["weights"]) for i, w in enumerate(row)),
            key=lambda e: abs(e[2]),
        )
        for layer in layers[1:]
    ]
    line_scale = min(1.0, max(0.35, 300 / max(1, *(len(e) for e in edges))))
    scales = [_brightness_scale(layer["values"]) for layer in layers]

    parts: list[str] = []
    add = parts.append

    # Static base: faint weights, dark neurons, headings.
    for k in range(1, L):
        add('<g stroke-linecap="round">')
        for i, j, w in edges[k]:
            a = abs(w) / wmax[k]
            (x1, y1), (x2, y2) = cols[k - 1][i], cols[k][j]
            add(
                f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" '
                f'stroke="{_rgb(BLUE if w > 0 else RED)}" stroke-opacity="{0.03 + 0.3 * a**1.5:.3f}" '
                f'stroke-width="{(0.35 + 2.6 * a) * line_scale:.2f}"/>'
            )
        add("</g>")

    for s_idx, s in enumerate(samples):
        t_start = s_idx * period
        t_end = t_start + period
        fade_out = (t_end - T_GAP, t_end)
        if img_box:
            x, y, w, h = img_box
            px = base64.b64decode(image["data"][s])
            href = _png(px, image["w"], image["h"], image["c"])
            cls = tl.fade(t_start, t_start + T_IN * 0.6, *fade_out)
            add(
                f'<image class="{cls}" x="{_f(x)}" y="{_f(y)}" width="{_f(w)}" height="{_f(h)}" '
                f'href="{href}" preserveAspectRatio="none" style="image-rendering:pixelated"/>'
            )
        for k in range(1, L):
            prev = layers[k - 1]["values"][s]
            contribs = [abs(w * prev[i]) for i, _, w in edges[k]]
            cmax = max(contribs, default=0) or 1e-9
            ts = t_start + starts[k]
            lit = tl.fade(ts + T_EDGE * 0.9, ts + T_EDGE * 1.1, *fade_out)
            add(f'<g class="{lit}" stroke-linecap="round">')
            for (i, j, w), c in zip(edges[k], contribs, strict=True):
                alpha = 0.55 * (c / cmax) ** 2
                if alpha < 0.04:
                    continue
                (x1, y1), (x2, y2) = cols[k - 1][i], cols[k][j]
                add(
                    f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" '
                    f'stroke="{_rgb(BLUE if w > 0 else RED)}" stroke-opacity="{alpha:.3f}" '
                    f'stroke-width="{(0.35 + 2.6 * abs(w) / wmax[k]) * line_scale:.2f}"/>'
                )
            add("</g>")
            comet = tl.add(
                [
                    (ts, "opacity:0;stroke-dashoffset:0.28"),
                    (ts + 0.01, "opacity:1;stroke-dashoffset:0.28"),
                    (ts + T_EDGE, "opacity:1;stroke-dashoffset:-0.72"),
                    (ts + T_EDGE * 1.2, "opacity:0;stroke-dashoffset:-0.92"),
                ]
            )
            add(f'<g class="{comet}" stroke-linecap="round" stroke-dasharray="0.28 2">')
            for (i, j, w), c in zip(edges[k], contribs, strict=True):
                if c / cmax < 0.1:
                    continue
                (x1, y1), (x2, y2) = cols[k - 1][i], cols[k][j]
                add(
                    f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" pathLength="1" '
                    f'stroke="{_rgb(BLUE if w > 0 else RED)}" stroke-opacity="{c / cmax:.2f}" '
                    f'stroke-width="{(1 + 2.8 * c / cmax) * max(0.6, line_scale):.2f}"/>'
                )
            add("</g>")

    # Neurons: dark base, then a lit layer per sample and layer that fades in as it fires.
    add('<g stroke="none">')
    for k in range(L):
        for x, y in cols[k]:
            add(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(r)}" fill="#0e0e0e"/>')
    add("</g>")
    for s_idx, s in enumerate(samples):
        t_start = s_idx * period
        fade_out = (t_start + period - T_GAP, t_start + period)
        for k in range(L):
            fire = (
                (t_start, t_start + T_IN)
                if k == 0
                else (
                    t_start + starts[k] + T_EDGE,
                    t_start + starts[k] + T_EDGE + T_FIRE,
                )
            )
            cls = tl.fade(*fire, *fade_out)
            add(f'<g class="{cls}" filter="url(#glow)">')
            for j, (x, y) in enumerate(cols[k]):
                v = layers[k]["values"][s][j]
                if k == L - 1 and out:
                    b = out["probs"][s][j]
                else:
                    b = min(1.0, abs(v) / scales[k])
                if b < 0.02:
                    continue
                color = (
                    (14 + 236 * b, 14 + 96 * b, 14 + 76 * b)
                    if v < 0 and not (k == L - 1 and out)
                    else (14 + 241 * b,) * 3
                )
                add(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(r)}" fill="{_rgb(color)}"/>')
            add("</g>")
    add('<g fill="url(#shade)" stroke="#ffffff" stroke-opacity="0.45">')
    for k in range(L):
        for x, y in cols[k]:
            add(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="{_f(r)}"/>')
    add("</g>")

    # Text: headings, layer details, hidden-neuron counts, output labels.
    head_size = 13
    add(f'<g font-family="{FONT}" text-anchor="middle">')
    for k, layer in enumerate(layers):
        heading = "Input" if k == 0 else "Output" if k == L - 1 else f"Hidden {k}"
        x = xs[k]
        y = cols[k][0][1] - r - 30
        add(
            f'<text x="{_f(x)}" y="{_f(y)}" fill="{TEXT}" font-size="{head_size}" font-weight="600">{heading}</text>'
        )
        add(
            f'<text x="{_f(x)}" y="{_f(y + 15)}" fill="{MUTED}" font-size="11">{escape(layer["detail"])}</text>'
        )
        if layer["size"] > len(layer["shown"]):
            more = layer["size"] - len(layer["shown"])
            add(
                f'<text x="{_f(x)}" y="{_f(cols[k][-1][1] + r + 18)}" fill="{MUTED}" font-size="11">+{more} more</text>'
            )
    add("</g>")
    if out:
        k = L - 1
        size = min(16.0, max(10.0, r * 1.25))
        add(f'<g font-family="{FONT}" font-size="{_f(size)}">')
        for j, (x, y) in enumerate(cols[k]):
            add(
                f'<text x="{_f(x + r + 8)}" y="{_f(y + 4)}" fill="#b9bec6">{escape(out["labels"][j])}</text>'
            )
        add("</g>")
        for s_idx, s in enumerate(samples):
            t_start = s_idx * period
            fire = t_start + starts[k] + T_EDGE
            cls = tl.fade(fire, fire + T_FIRE, t_start + period - T_GAP, t_start + period)
            pred = out["predicted"][s]
            add(f'<g class="{cls}" font-family="{FONT}">')
            for j, (x, y) in enumerate(cols[k]):
                label = escape(out["labels"][j])
                if layers[k]["shown"][j] == pred:
                    add(
                        f'<rect x="{_f(x + r + 4)}" y="{_f(y - size * 0.75)}" width="{_f(size * 0.62 * len(label) + 8)}" '
                        f'height="{_f(size * 1.1)}" rx="3" fill="{BG}"/>'
                        f'<text x="{_f(x + r + 8)}" y="{_f(y + 4)}" fill="#fff" font-weight="700" '
                        f'font-size="{_f(size)}">{label}</text>'
                    )
                pct = f"{out['probs'][s][j] * 100:.0f}%"
                add(
                    f'<text x="{_f(x + r + 14 + size * 0.62 * len(label))}" y="{_f(y + 4)}" '
                    f'fill="{MUTED}" font-size="11">{pct}</text>'
                )
            add("</g>")
    if img_box:
        x, y, w, h = img_box
        add(
            f'<rect x="{_f(x)}" y="{_f(y)}" width="{_f(w)}" height="{_f(h)}" fill="none" stroke="#ffffff" stroke-opacity="0.35"/>'
        )
        if image["h"] <= 64 and w / image["w"] >= 4:
            lines = [
                f"M{_f(x + i * w / image['w'])} {_f(y)}v{_f(h)}" for i in range(1, image["w"])
            ] + [f"M{_f(x)} {_f(y + i * h / image['h'])}h{_f(w)}" for i in range(1, image["h"])]
            add(f'<path d="{"".join(lines)}" stroke="#ffffff" stroke-opacity="0.09" fill="none"/>')

    defs = (
        '<radialGradient id="shade" cx="0.35" cy="0.3" r="0.75">'
        '<stop offset="0" stop-color="#fff" stop-opacity="0.35"/>'
        '<stop offset="0.55" stop-color="#fff" stop-opacity="0"/>'
        '<stop offset="1" stop-color="#000" stop-opacity="0.35"/></radialGradient>'
        f'<filter id="glow" x="-50%" y="-50%" width="200%" height="200%">'
        f'<feGaussianBlur stdDeviation="{_f(r * 0.6)}" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>'
    )
    return _document(payload, parts, defs, tl, width, height, title)


def _document(
    payload: dict[str, Any],
    parts: list[str],
    defs: str,
    tl: _Timeline,
    width: int,
    height: int,
    title: bool,
) -> str:
    header = ""
    if title:
        m = payload["model"]
        meta = f"{m['name']} · {_count(m['params'])} parameters"
        header = (
            f'<text x="20" y="32" font-family="{FONT}" font-size="17" font-weight="600" fill="{TEXT}">'
            f"{escape(payload['title'])}</text>"
            f'<text x="20" y="52" font-family="{FONT}" font-size="12" fill="{MUTED}">{escape(meta)}</text>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{escape(payload["title"], {chr(34): "&quot;"})}">'
        f"<title>{escape(payload['title'])}</title>"
        f"<defs>{defs}</defs><style>{tl.css()}</style>"
        f'<rect width="{width}" height="{height}" rx="12" fill="{BG}"/>'
        f"{header}{''.join(parts)}</svg>"
    )


def _count(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return str(n)


def render_svg(
    payload: dict[str, Any],
    *,
    samples: Sequence[int] | None = None,
    width: int = 960,
    height: int = 540,
    title: bool = True,
) -> str:
    """Render ``payload`` as an animated SVG.

    Args:
        payload: From ``export.build_payload``.
        samples: Indices of the samples to animate, one after another. Defaults to the
            first three.
        width: Width in pixels.
        height: Height in pixels.
        title: Draw the title and parameter count in the top-left corner.

    Returns:
        The SVG document as a string.
    """
    count = payload["samples"]
    chosen = list(samples) if samples is not None else list(range(min(3, count)))
    if not chosen or any(not 0 <= s < count for s in chosen):
        raise ValueError(f"samples must be indices in [0, {count}), got {chosen}.")
    if payload["view"] == "neurons":
        return neuron_svg(payload, chosen, width, height, title)
    from torchscope.svg_slabs import slab_svg

    return slab_svg(payload, chosen, width, height, title)
