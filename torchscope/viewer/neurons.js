// ---- Neuron view: every neuron and weight of a small MLP, drawn on a 2D canvas ----

function createNeuronView(root, P, ui) {
  const N = P.neurons;
  const layers = N.layers;
  const L = layers.length;
  const out = N.output;

  const canvas = document.createElement("canvas");
  canvas.className = "view-canvas";
  root.appendChild(canvas);
  const ctx = canvas.getContext("2d");

  let sample = 0;
  let hover = null; // {k, j}: layer index, index into layer.shown
  let geo = null;
  let lastT = -1;
  let dirty = true;
  let W = 0;
  let H = 0;

  // Timeline: the input fades in, then each layer receives a pulse along its edges and fires.
  const T_IN = 0.8;
  const T_EDGE = 1.1;
  const T_FIRE = 0.35;
  const T_HOLD = 0.6;
  const starts = layers.map((_, k) => (k === 0 ? 0 : T_IN + (k - 1) * (T_EDGE + T_FIRE)));
  const duration = T_IN + (L - 1) * (T_EDGE + T_FIRE) + T_HOLD;

  // Brightness scale per layer: a high percentile of the non-zero |activations|, so a
  // single large value (common with ReLU) does not leave every other neuron dark.
  const scale = layers.map((layer) => {
    const mags = [];
    let neg = false;
    for (const row of layer.values)
      for (const v of row) {
        if (Math.abs(v) > 1e-9) mags.push(Math.abs(v));
        if (v < -1e-6) neg = true;
      }
    mags.sort((a, b) => a - b);
    const max = mags.length ? mags[Math.min(mags.length - 1, Math.floor(mags.length * 0.9))] : 1;
    let wmax = 0;
    for (const row of layer.weights || []) for (const w of row) wmax = Math.max(wmax, Math.abs(w));
    return { max: max || 1, neg, wmax: wmax || 1 };
  });
  const isOutput = (k) => k === L - 1 && !!out;
  const value = (k, j) => layers[k].values[sample][j];
  const brightness = (k, j) =>
    isOutput(k) ? out.probs[sample][j] : Math.min(1, Math.abs(value(k, j)) / scale[k].max);
  const level = (k, t) =>
    k === 0 ? smoothstep(0, T_IN, t) : smoothstep(starts[k] + T_EDGE, starts[k] + T_EDGE + T_FIRE, t);
  const heading = (k) => (k === 0 ? "Input" : k === L - 1 ? "Output" : `Hidden ${k}`);

  // One step per column: the input appearing, then each layer receiving its signals and firing.
  const predText = () => {
    if (!out) return "";
    const j = layers[L - 1].shown.indexOf(out.predicted[sample]);
    const p = j >= 0 ? out.probs[sample][j] : null;
    return ` Prediction: <b>${esc(out.labels[j] ?? out.predicted[sample])}</b>${p != null ? ` (${(p * 100).toFixed(0)}%)` : ""}.`;
  };
  const stages = layers.map((layer, k) => ({
    start: starts[k],
    end: k === 0 ? T_IN : starts[k] + T_EDGE + T_FIRE,
    title: heading(k),
    text: () => esc(layer.explain || "") + (k === L - 1 ? predText() : ""),
  }));

  const images = N.image ? Array.from({ length: P.samples }, (_, s) => imageToCanvas(N.image, s)) : null;

  ui.legend.innerHTML =
    `<div><span class="sw" style="background:${rgba(COLORS.blue, 1)}"></span>positive weight</div>` +
    `<div><span class="sw" style="background:${rgba(COLORS.red, 1)}"></span>negative weight</div>` +
    `<div class="muted">brightness = activation</div>`;

  function layout() {
    const rect = root.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    W = rect.width;
    H = rect.height;
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
    canvas.style.width = `${W}px`;
    canvas.style.height = `${H}px`;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const narrow = W < 640;
    const top = narrow ? 150 : 110;
    // Leave room for the control bar, whose height changes with the caption.
    const bottom = (document.getElementById("controls")?.offsetHeight || 120) + (narrow ? 50 : 40);
    const availH = Math.max(120, H - top - bottom);
    const maxShown = Math.max(...layers.map((l) => l.shown.length));
    const r = clamp(availH / (maxShown * 2.6), 2.5, 15);

    let img = null;
    if (images) {
      const size = Math.min(availH * 0.62, W * (narrow ? 0.2 : 0.16));
      const aspect = N.image.w / N.image.h;
      const w = aspect >= 1 ? size : size * aspect;
      const h = aspect >= 1 ? size / aspect : size;
      img = { x: narrow ? 12 : 28, y: top + availH / 2 - h / 2, w, h };
    }
    const left = (img ? img.x + img.w + Math.max(28, W * 0.05) : narrow ? 20 : 40) + r;
    const right = W - (out ? (narrow ? 64 : 120) : 30) - r;
    const xs = layers.map((_, k) => (L === 1 ? left : lerp(left, right, k / (L - 1))));
    const cols = layers.map((layer, k) => {
      const n = layer.shown.length;
      const gap = Math.min(r * 2.7, availH / Math.max(1, n));
      const y0 = top + availH / 2 - (gap * (n - 1)) / 2;
      return layer.shown.map((_, j) => ({ x: xs[k], y: y0 + j * gap }));
    });
    // Draw weak edges first so strong ones end up on top.
    const edges = layers.map((layer, k) => {
      if (!k) return [];
      const list = [];
      layer.weights.forEach((row, j) => row.forEach((w, i) => list.push({ i, j, w, a: Math.abs(w) / scale[k].wmax })));
      return list.sort((a, b) => a.a - b.a);
    });
    const edgeCount = Math.max(1, ...edges.map((e) => e.length));
    geo = { r, cols, xs, edges, top, availH, img, narrow, lineScale: clamp(300 / edgeCount, 0.35, 1) };
    dirty = true;
  }

  const touches = (k, e) =>
    hover && ((hover.k === k && hover.j === e.j) || (hover.k === k - 1 && hover.j === e.i));

  function drawEdges(k, t) {
    const src = geo.cols[k - 1];
    const dst = geo.cols[k];
    const p = (t - starts[k]) / T_EDGE;
    const a = layers[k - 1].shown.map((_, i) => value(k - 1, i));
    let cmax = 1e-9;
    for (const e of geo.edges[k]) cmax = Math.max(cmax, Math.abs(e.w * a[e.i]));
    ctx.lineCap = "round";
    for (const e of geo.edges[k]) {
      const contrib = Math.abs(e.w * a[e.i]) / cmax;
      let alpha = 0.03 + 0.3 * Math.pow(e.a, 1.5);
      if (p >= 1) alpha += 0.55 * contrib * contrib;
      if (hover) alpha = touches(k, e) ? Math.max(alpha, 0.25 + 0.65 * e.a) : alpha * 0.1;
      ctx.strokeStyle = rgba(e.w > 0 ? COLORS.blue : COLORS.red, Math.min(1, alpha));
      ctx.lineWidth = (0.35 + 2.6 * e.a) * geo.lineScale;
      ctx.beginPath();
      ctx.moveTo(src[e.i].x, src[e.i].y);
      ctx.lineTo(dst[e.j].x, dst[e.j].y);
      ctx.stroke();
    }
    // Pulses: bright comets travelling along each edge, weighted by w * a.
    if (p > 0 && p < 1.2) {
      const fade = p > 1 ? 1 - (p - 1) / 0.2 : 1;
      const head = clamp(p, 0, 1);
      const tail = clamp(p - 0.28, 0, 1);
      ctx.globalCompositeOperation = "lighter";
      for (const e of geo.edges[k]) {
        const contrib = Math.abs(e.w * a[e.i]) / cmax;
        if (contrib < 0.06 || (hover && !touches(k, e))) continue;
        const s = src[e.i];
        const d = dst[e.j];
        ctx.strokeStyle = rgba(e.w > 0 ? COLORS.blue : COLORS.red, contrib * fade);
        ctx.lineWidth = (1 + 2.8 * contrib) * Math.max(0.6, geo.lineScale);
        ctx.beginPath();
        ctx.moveTo(lerp(s.x, d.x, tail), lerp(s.y, d.y, tail));
        ctx.lineTo(lerp(s.x, d.x, head), lerp(s.y, d.y, head));
        ctx.stroke();
      }
      ctx.globalCompositeOperation = "source-over";
    }
  }

  function drawNeuron(x, y, r, v, negative, highlighted) {
    const c = negative ? [lerp(14, 250, v), lerp(14, 110, v), lerp(14, 90, v)] : [lerp(14, 255, v), lerp(14, 255, v), lerp(14, 255, v)];
    if (v > 0.12) {
      ctx.shadowColor = negative ? rgba(COLORS.red, 0.8) : "rgba(255,255,255,0.7)";
      ctx.shadowBlur = r * 1.8 * v;
    }
    ctx.fillStyle = `rgb(${c.map(Math.round).join(",")})`;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    if (r > 4) {
      // Soft highlight and shadow so neurons read as spheres.
      const g = ctx.createRadialGradient(x - r * 0.35, y - r * 0.4, r * 0.1, x, y, r);
      g.addColorStop(0, "rgba(255,255,255,0.35)");
      g.addColorStop(0.55, "rgba(255,255,255,0)");
      g.addColorStop(1, "rgba(0,0,0,0.35)");
      ctx.fillStyle = g;
      ctx.fill();
    }
    ctx.lineWidth = highlighted ? 2 : 1;
    ctx.strokeStyle = highlighted ? rgba(COLORS.accent, 1) : "rgba(255,255,255,0.45)";
    ctx.stroke();
  }

  function pixelOf(index) {
    const { c, h, w } = N.image;
    const plane = h * w;
    const rem = index % plane;
    return { ch: Math.floor(index / plane) % c, x: rem % w, y: Math.floor(rem / w) };
  }

  function drawImage(t) {
    const g = geo.img;
    const a = level(0, t);
    ctx.save();
    ctx.globalAlpha = 0.25 + 0.75 * a;
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(images[sample], g.x, g.y, g.w, g.h);
    ctx.globalAlpha = 1;
    const { h, w } = N.image;
    if (h <= 64 && g.w / w >= 4) {
      ctx.strokeStyle = "rgba(255,255,255,0.09)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = 0; x <= w; x++) {
        ctx.moveTo(g.x + (x * g.w) / w, g.y);
        ctx.lineTo(g.x + (x * g.w) / w, g.y + g.h);
      }
      for (let y = 0; y <= h; y++) {
        ctx.moveTo(g.x, g.y + (y * g.h) / h);
        ctx.lineTo(g.x + g.w, g.y + (y * g.h) / h);
      }
      ctx.stroke();
    }
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.strokeRect(g.x, g.y, g.w, g.h);
    if (hover && hover.k === 0) {
      const px = pixelOf(layers[0].shown[hover.j]);
      const cw = g.w / w;
      const chh = g.h / h;
      const cx = g.x + px.x * cw;
      const cy = g.y + px.y * chh;
      ctx.strokeStyle = rgba(COLORS.accent, 1);
      ctx.lineWidth = 2;
      ctx.strokeRect(cx, cy, cw, chh);
      const n = geo.cols[0][hover.j];
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(cx + cw / 2, cy + chh / 2);
      ctx.lineTo(n.x, n.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
    ctx.restore();
  }

  function draw(t) {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, W, H);
    if (geo.img) drawImage(t);
    for (let k = 1; k < L; k++) drawEdges(k, t);

    const r = geo.r;
    for (let k = 0; k < L; k++) {
      const lit = level(k, t);
      const col = geo.cols[k];
      col.forEach((pos, j) => {
        const neg = scale[k].neg && value(k, j) < 0 && !isOutput(k);
        const isHover = hover && hover.k === k && hover.j === j;
        drawNeuron(pos.x, pos.y, r, brightness(k, j) * lit, neg, isHover);
      });

      // Column heading, detail and how many neurons are hidden.
      const layer = layers[k];
      const x = geo.xs[k];
      ctx.textAlign = "center";
      ctx.fillStyle = "#eceef1";
      ctx.font = `600 ${geo.narrow ? 11 : 13}px system-ui, sans-serif`;
      const headY = col[0].y - r - (geo.narrow ? 26 : 30);
      ctx.fillText(heading(k), x, headY);
      if (!geo.narrow) {
        ctx.fillStyle = "#8d939c";
        ctx.font = "11px system-ui, sans-serif";
        ctx.fillText(layer.detail, x, headY + 15);
      }
      if (layer.size > layer.shown.length) {
        ctx.fillStyle = "#8d939c";
        ctx.font = "11px system-ui, sans-serif";
        ctx.fillText(`+${layer.size - layer.shown.length} more`, x, col[col.length - 1].y + r + 18);
      }
    }

    // Output labels and probabilities.
    if (out) {
      const k = L - 1;
      const lit = level(k, t);
      const pred = out.predicted[sample];
      const col = geo.cols[k];
      ctx.textAlign = "left";
      col.forEach((pos, j) => {
        const isPred = layers[k].shown[j] === pred && lit > 0.5;
        ctx.font = `${isPred ? 700 : 500} ${clamp(r * 1.25, 10, 16)}px system-ui, sans-serif`;
        ctx.fillStyle = isPred ? "#fff" : "#b9bec6";
        ctx.fillText(out.labels[j], pos.x + r + 8, pos.y + 4);
        if (lit > 0 && !geo.narrow) {
          const w = ctx.measureText(out.labels[j]).width;
          ctx.font = "11px system-ui, sans-serif";
          ctx.fillStyle = `rgba(141,147,156,${lit})`;
          ctx.fillText(`${(out.probs[sample][j] * 100).toFixed(0)}%`, pos.x + r + 14 + w, pos.y + 4);
        }
      });
    }
  }

  function onMove(e) {
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;
    let best = null;
    let bestD = (geo.r + 5) ** 2;
    geo.cols.forEach((col, k) =>
      col.forEach((p, j) => {
        const d = (p.x - mx) ** 2 + (p.y - my) ** 2;
        if (d < bestD) {
          bestD = d;
          best = { k, j };
        }
      }),
    );
    // Hovering a pixel of the input image selects its neuron, if that pixel is shown.
    if (!best && geo.img && mx >= geo.img.x && mx < geo.img.x + geo.img.w && my >= geo.img.y && my < geo.img.y + geo.img.h) {
      const px = Math.floor(((mx - geo.img.x) / geo.img.w) * N.image.w);
      const py = Math.floor(((my - geo.img.y) / geo.img.h) * N.image.h);
      const j = layers[0].shown.findIndex((idx) => {
        const p = pixelOf(idx);
        return p.x === px && p.y === py;
      });
      if (j >= 0) best = { k: 0, j };
    }
    if (JSON.stringify(best) !== JSON.stringify(hover)) {
      hover = best;
      dirty = true;
    }
    if (!hover) return ui.hideTip();
    ui.tip(neuronTip(hover.k, hover.j), e.clientX, e.clientY);
  }
  function sourceName(k, i) {
    if (k === 0 && N.image) {
      const p = pixelOf(i);
      return `pixel (${p.x}, ${p.y})`;
    }
    return `${heading(k).toLowerCase()} #${i}`;
  }

  /** Tooltip with the neuron's value and, for layers, the weighted sum that produced it. */
  function neuronTip(k, j) {
    const layer = layers[k];
    const idx = layer.shown[j];
    const v = value(k, j);
    let html = `<b>${heading(k)}</b> · neuron ${idx}`;
    if (k === 0) {
      html += `<br><span class="muted">value</span> ${fmtNum(v)}`;
      if (N.image) html += ` <span class="muted">from ${sourceName(0, idx)}</span>`;
      return html;
    }
    const terms = layer.terms[sample][j];
    const z = layer.z[sample][j];
    const b = layer.bias ? layer.bias[j] : 0;
    const top = terms.reduce((s, [, w, a]) => s + w * a, 0);
    const rest = z - b - top;
    const others = layers[k - 1].size - terms.length;
    const row = (sign, w, a, src) =>
      `<tr><td class="${sign < 0 ? "neg" : "pos"}">${sign < 0 ? "−" : "+"}</td><td class="num">${fmtNum(Math.abs(w))}</td><td>×</td><td class="num">${fmtNum(a)}</td><td class="src">${esc(src)}</td></tr>`;
    html += `<div class="math">z = Σ weight × input + bias<table>`;
    for (const [i, w, a] of terms) html += row(Math.sign(w * a) || 1, w, a, sourceName(k - 1, i));
    if (others > 0) html += `<tr><td class="${rest < 0 ? "neg" : "pos"}">${rest < 0 ? "−" : "+"}</td><td class="num">${fmtNum(Math.abs(rest))}</td><td></td><td></td><td class="src">${others} other inputs</td></tr>`;
    if (layer.bias) html += `<tr><td class="${b < 0 ? "neg" : "pos"}">${b < 0 ? "−" : "+"}</td><td class="num">${fmtNum(Math.abs(b))}</td><td></td><td></td><td class="src">bias</td></tr>`;
    html += `<tr class="total"><td>=</td><td class="num"><b>${fmtNum(z)}</b></td><td></td><td></td><td class="src">z</td></tr></table>`;
    if (isOutput(k)) {
      html += `softmax → <b>${(out.probs[sample][j] * 100).toFixed(1)}%</b> <span class="src">${esc(out.labels[j])}</span>`;
    } else if (layer.act && layer.exact) {
      html += `${esc(layer.act)}(${fmtNum(z)}) = <b>${fmtNum(v)}</b>`;
    } else if (layer.act || !layer.exact) {
      html += `after ${esc(layer.detail.split(" · ").slice(1).join(", "))}: <b>${fmtNum(v)}</b>`;
    } else {
      html += `a = z = <b>${fmtNum(v)}</b>`;
    }
    return html + "</div>";
  }

  function onLeave() {
    hover = null;
    dirty = true;
    ui.hideTip();
  }
  canvas.addEventListener("pointermove", onMove);
  canvas.addEventListener("pointerleave", onLeave);
  const ro = new ResizeObserver(layout);
  ro.observe(root);
  ro.observe(document.getElementById("controls"));
  layout();

  return {
    duration,
    stages,
    frame(t) {
      if (t === lastT && !dirty) return;
      lastT = t;
      dirty = false;
      draw(t);
    },
    setSample(s) {
      sample = s;
      dirty = true;
    },
    destroy() {
      ro.disconnect();
      canvas.remove();
    },
  };
}
