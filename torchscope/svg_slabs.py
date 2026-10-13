"""Animated SVG of the slab view: boxes in an orthographic projection, pure CSS animation."""

from __future__ import annotations

import base64
import math
from collections.abc import Sequence
from typing import Any
from xml.sax.saxutils import escape

from torchscope.svg import ACCENT, FONT, MUTED, TEXT, _document, _f, _png, _rgb, _Timeline

STEP, FADE, T0, T_HOLD, T_GAP = 0.5, 0.35, 0.3, 1.8, 0.5
GAP = 2.4
VIEW_DIR = (-1.0, 0.55, 0.95)  # same camera direction as the HTML slab view
SYMBOLS = {"add": "+", "sub": "−", "mul": "×", "div": "÷", "matmul": "@"}


def _lut() -> list[tuple[int, int, int]]:
    stops = [
        (0, (4, 8, 15)),
        (0.3, (16, 52, 92)),
        (0.6, (40, 140, 200)),
        (0.82, (110, 210, 240)),
        (1, (255, 255, 255)),
    ]
    out = []
    for i in range(256):
        t = i / 255
        k = 0
        while k < len(stops) - 2 and t > stops[k + 1][0]:
            k += 1
        (t0, c0), (t1, c1) = stops[k], stops[k + 1]
        u = (t - t0) / (t1 - t0)
        out.append(tuple(round(a + (b - a) * u) for a, b in zip(c0, c1, strict=True)))
    return out


LUT = _lut()


def _face_png(face: dict[str, Any], sample: int) -> str:
    raw = base64.b64decode(face["data"][sample])
    return _png(bytes(c for v in raw for c in LUT[v]), face["w"], face["h"], 3)


def _image_png(img: dict[str, Any], sample: int) -> str:
    return _png(base64.b64decode(img["data"][sample]), img["w"], img["h"], img["c"])


def _sx(d: int) -> float:
    return 0.3 + 0.42 * math.log2(d + 1)


def _sy(d: int) -> float:
    return 0.4 + 0.72 * math.log2(d + 1)


def _layout(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rank nodes by longest path, keep the longest chain on lane 0, branches behind."""
    items = []
    for n in nodes:
        glyph = n["kind"] == "op" or n["role"] == "reshape"
        c, h, w = n.get("dims", [1, 1, 1])
        size = (0.9, 0.9, 0.9) if glyph else (_sx(c), _sy(h), _sy(w))
        items.append({"node": n, "glyph": glyph, "size": size, "preds": n["inputs"], "succs": []})
    for it in items:
        for p in it["preds"]:
            items[p]["succs"].append(it["node"]["id"])
    for it in items:
        it["rank"] = max((items[p]["rank"] + 1 for p in it["preds"]), default=0)
    for it in reversed(items):
        it["down"] = max((items[s]["down"] + 1 for s in it["succs"]), default=0)
    cur = max((it for it in items if not it["preds"]), key=lambda it: it["down"], default=None)
    while cur is not None:
        cur["main"] = True
        nxt = [items[s] for s in cur["succs"]]
        cur = max(nxt, key=lambda it: it["down"]) if nxt else None
    lane_gap = max(it["size"][2] for it in items) + 2.5
    by_rank: dict[int, list[dict[str, Any]]] = {}
    for it in items:
        by_rank.setdefault(it["rank"], []).append(it)
    x = 0.0
    for i, r in enumerate(sorted(by_rank)):
        row = sorted(by_rank[r], key=lambda it: (not it.get("main"), it["node"]["id"]))
        width = max(it["size"][0] for it in row)
        if i:
            x += width / 2
        off = 0
        for it in row:
            if it.get("main"):
                lane = 0
            else:
                off = -(off + 1) if off >= 0 else -off
                lane = off
            it["lane"], it["x"], it["z"] = lane, x, lane * lane_gap
        x += width / 2 + GAP
    return items


class _Camera:
    """Orthographic projection looking along ``-VIEW_DIR``."""

    def __init__(self) -> None:
        d = VIEW_DIR
        n = math.sqrt(sum(v * v for v in d))
        f = tuple(-v / n for v in d)
        right = (-f[2], 0.0, f[0])
        rn = math.sqrt(sum(v * v for v in right))
        self.right = tuple(v / rn for v in right)
        r = self.right
        self.up = (r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0])
        self.forward = f
        self.scale, self.ox, self.oy = 1.0, 0.0, 0.0

    def raw(self, p: Sequence[float]) -> tuple[float, float]:
        return (
            sum(a * b for a, b in zip(p, self.right, strict=True)),
            -sum(a * b for a, b in zip(p, self.up, strict=True)),
        )

    def depth(self, p: Sequence[float]) -> float:
        return sum(a * b for a, b in zip(p, self.forward, strict=True))

    def __call__(self, p: Sequence[float]) -> tuple[float, float]:
        x, y = self.raw(p)
        return x * self.scale + self.ox, y * self.scale + self.oy


def _faces(it: dict[str, Any]) -> list[tuple[str, tuple, tuple, tuple]]:
    """Visible faces as (texture, top-left, top-right, bottom-left) corners in 3D."""
    sx, sy, sz = (v / 2 for v in it["size"])
    x, z = it["x"], it["z"]
    return [
        ("front", (x - sx, sy, z - sz), (x - sx, sy, z + sz), (x - sx, -sy, z - sz)),
        ("side", (x - sx, sy, z + sz), (x + sx, sy, z + sz), (x - sx, -sy, z + sz)),
        ("top", (x - sx, sy, z - sz), (x + sx, sy, z - sz), (x - sx, sy, z + sz)),
    ]


def _quad(cam: _Camera, a: tuple, b: tuple, c: tuple) -> list[tuple[float, float]]:
    pa, pb, pc = cam(a), cam(b), cam(c)
    pd = (pb[0] + pc[0] - pa[0], pb[1] + pc[1] - pa[1])
    return [pa, pb, pd, pc]


def _points(q: list[tuple[float, float]]) -> str:
    return " ".join(f"{_f(x)},{_f(y)}" for x, y in q)


def slab_svg(
    payload: dict[str, Any],
    samples: Sequence[int],
    width: int,
    height: int,
    title: bool,
) -> str:
    """Render the slab view as an animated SVG that cycles through ``samples``."""
    nodes = payload["graph"]["nodes"]
    items = _layout(nodes)
    max_rank = max(it["rank"] for it in items)
    period = T0 + max_rank * STEP + FADE + T_HOLD + T_GAP
    tl = _Timeline(period * len(samples))
    cam = _Camera()

    # Fit the projected scene into the viewport.
    pts = []
    for it in items:
        sx, sy, sz = (v / 2 for v in it["size"])
        for dx in (-sx, sx):
            for dy in (-sy, sy):
                for dz in (-sz, sz):
                    pts.append(cam.raw((it["x"] + dx, dy, it["z"] + dz)))
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    top, bottom, side = (78 if title else 24), 56, 30
    scale = min(
        (width - 2 * side) / (max(xs) - min(xs)), (height - top - bottom) / (max(ys) - min(ys))
    )
    cam.scale = scale
    cam.ox = side + (width - 2 * side - (max(xs) - min(xs)) * scale) / 2 - min(xs) * scale
    cam.oy = top + (height - top - bottom - (max(ys) - min(ys)) * scale) / 2 - min(ys) * scale

    parts: list[str] = []
    add = parts.append

    # Edges with a pulse travelling along each one.
    for it in items:
        for p in it["preds"]:
            src = items[p]
            a = (src["x"] + src["size"][0] / 2, 0.0, src["z"])
            b = (it["x"] - it["size"][0] / 2, 0.0, it["z"])
            if it["rank"] - src["rank"] > 1 and src["lane"] == it["lane"]:
                h = (
                    max(
                        o["size"][1] / 2
                        for o in items
                        if src["rank"] <= o["rank"] <= it["rank"] and o["lane"] == it["lane"]
                    )
                    + 1.4
                )
                a = (src["x"], src["size"][1] / 2, src["z"])
                b = (it["x"], it["size"][1] / 2, it["z"])
                ctrl = [a, (a[0] + 0.6, h + 0.6, a[2]), (b[0] - 0.6, h + 0.6, b[2]), b]
            else:
                dx = (b[0] - a[0]) / 2
                ctrl = [a, (a[0] + dx, 0, a[2]), (b[0] - dx, 0, b[2]), b]
            curve = [cam(_bezier(ctrl, t / 16)) for t in range(17)]
            d = "M" + " L".join(f"{_f(x)} {_f(y)}" for x, y in curve)
            add(f'<path d="{d}" fill="none" stroke="#1d3440" stroke-width="1.5"/>')
            for s_idx in range(len(samples)):
                base = s_idx * period
                t_a = base + T0 + src["rank"] * STEP + FADE * 0.5
                t_b = base + T0 + it["rank"] * STEP
                lit = tl.fade(t_b - 0.05, t_b + 0.3, base + period - T_GAP, base + period)
                add(
                    f'<path class="{lit}" d="{d}" fill="none" stroke="{_rgb(ACCENT)}" stroke-opacity="0.7" stroke-width="1.5"/>'
                )
                frames = [
                    (t_a, f"opacity:0;transform:translate({_f(curve[0][0])}px,{_f(curve[0][1])}px)")
                ]
                for i, (x, y) in enumerate(curve):
                    t = t_a + (t_b - t_a) * i / (len(curve) - 1)
                    frames.append((t, f"opacity:1;transform:translate({_f(x)}px,{_f(y)}px)"))
                frames.append(
                    (
                        t_b + 0.01,
                        f"opacity:0;transform:translate({_f(curve[-1][0])}px,{_f(curve[-1][1])}px)",
                    )
                )
                cls = tl.add(frames)
                add(
                    f'<circle class="{cls}" r="{_f(max(2.0, scale * 0.14))}" fill="#9fe8ff" opacity="0"/>'
                )

    # Boxes, back to front.
    order = sorted(items, key=lambda it: -cam.depth((it["x"], 0, it["z"])))
    for it in order:
        n = it["node"]
        t_lit = T0 + it["rank"] * STEP
        if it["glyph"]:
            cx, cy = cam((it["x"], 0, it["z"]))
            rad = max(4.0, 0.45 * scale)
            add(f'<circle cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(rad)}" fill="#15232b"/>')
            for s_idx in range(len(samples)):
                base = s_idx * period
                cls = tl.fade(
                    base + t_lit, base + t_lit + FADE, base + period - T_GAP, base + period
                )
                add(
                    f'<circle class="{cls}" cx="{_f(cx)}" cy="{_f(cy)}" r="{_f(rad)}" fill="{_rgb(ACCENT)}"/>'
                )
            symbol = SYMBOLS.get(n["type"])
            if symbol:
                add(
                    f'<text x="{_f(cx)}" y="{_f(cy + rad * 0.45)}" text-anchor="middle" font-family="{FONT}" '
                    f'font-size="{_f(rad * 1.3)}" font-weight="700" fill="#000">{symbol}</text>'
                )
            continue
        faces = _faces(it)
        for s_idx, s in enumerate(samples):
            base = s_idx * period
            visible = tl.fade(base, base + 0.01, base + period - T_GAP, base + period)
            add(f'<g class="{visible}">')
            for name, a, b, c in faces:
                if name == "front" and "image" in n:
                    href, tw, th = _image_png(n["image"], s), n["image"]["w"], n["image"]["h"]
                else:
                    f = n["faces"][name]
                    href, tw, th = _face_png(f, s), f["w"], f["h"]
                pa, pb, pc = cam(a), cam(b), cam(c)
                m = (
                    (pb[0] - pa[0]) / tw, (pb[1] - pa[1]) / tw,
                    (pc[0] - pa[0]) / th, (pc[1] - pa[1]) / th,
                    pa[0], pa[1],
                )  # fmt: skip
                add(
                    f'<image width="{tw}" height="{th}" href="{href}" preserveAspectRatio="none" '
                    f'transform="matrix({",".join(f"{v:.4f}" for v in m)})" style="image-rendering:pixelated"/>'
                )
            # Dark veil that lifts when the pulse reaches this layer, with a short flash.
            veil = tl.add(
                [
                    (base + t_lit, "opacity:0.88"),
                    (base + t_lit + FADE, "opacity:0"),
                ]
            )
            flash = tl.add(
                [
                    (base + t_lit, "opacity:0"),
                    (base + t_lit + FADE * 0.6, "opacity:0.35"),
                    (base + t_lit + FADE * 1.8, "opacity:0"),
                ]
            )
            for _, a, b, c in faces:
                q = _points(_quad(cam, a, b, c))
                add(f'<polygon class="{veil}" points="{q}" fill="#000"/>')
                add(f'<polygon class="{flash}" points="{q}" fill="#9fe8ff" opacity="0"/>')
            add("</g>")
        for _, a, b, c in faces:
            add(
                f'<polygon points="{_points(_quad(cam, a, b, c))}" fill="none" stroke="#fff" stroke-opacity="0.35" stroke-width="0.8"/>'
            )

    # Labels under each box, and the prediction for each sample.
    if len(items) <= 30:
        add(f'<g font-family="{FONT}" text-anchor="middle">')
        for it in items:
            n = it["node"]
            if it["glyph"] and SYMBOLS.get(n["type"]):
                continue
            sx, sy, sz = (v / 2 for v in it["size"])
            x, y = cam((it["x"], -sy, it["z"] + sz))
            add(
                f'<text x="{_f(x)}" y="{_f(y + 15)}" fill="{TEXT}" font-size="11">{escape(n["label"])}</text>'
            )
            shape = "×".join(str(v) for v in n["shape"]) or "scalar"
            add(f'<text x="{_f(x)}" y="{_f(y + 28)}" fill="{MUTED}" font-size="10">{shape}</text>')
        add("</g>")
    out = next((it for it in items if it["node"].get("top")), None)
    if out:
        for s_idx, s in enumerate(samples):
            base = s_idx * period
            t_lit = base + T0 + out["rank"] * STEP + FADE
            cls = tl.fade(t_lit, t_lit + 0.3, base + period - T_GAP, base + period)
            label, p = out["node"]["top"][s][0]
            add(
                f'<text class="{cls}" x="{width - 24}" y="{height - 22}" text-anchor="end" font-family="{FONT}" '
                f'font-size="14" fill="{TEXT}">prediction <tspan font-weight="700">{escape(label)}</tspan> '
                f'<tspan fill="{MUTED}">{p * 100:.0f}%</tspan></text>'
            )
    return _document(payload, parts, "", tl, width, height, title)


def _bezier(ctrl: list[tuple], t: float) -> tuple[float, float, float]:
    u = 1 - t
    w = (u**3, 3 * u * u * t, 3 * u * t * t, t**3)
    return tuple(sum(wi * p[k] for wi, p in zip(w, ctrl, strict=True)) for k in range(3))
