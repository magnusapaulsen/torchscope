// ---- Learning widgets for the details panel: feature-map grids, kernels, convolution ----

const ACTIVATION_FNS = {
  relu: (z) => Math.max(0, z),
  relu6: (z) => clamp(z, 0, 6),
  leakyrelu: (z) => (z > 0 ? z : 0.01 * z),
  sigmoid: (z) => 1 / (1 + Math.exp(-z)),
  tanh: (z) => Math.tanh(z),
  gelu: (z) => 0.5 * z * (1 + Math.tanh(0.7978845608 * (z + 0.044715 * z ** 3))),
  silu: (z) => z / (1 + Math.exp(-z)),
};
const activationFn = (name) => (name ? ACTIVATION_FNS[name.toLowerCase()] : null);

/** One channel of a `{c, h, w, data}` channel stack, drawn with the LUT. */
function channelCanvas(ch, sample, c) {
  const canvas = document.createElement("canvas");
  canvas.width = ch.w;
  canvas.height = ch.h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(ch.w, ch.h);
  const src = bytes(ch.data[sample]);
  const off = c * ch.h * ch.w;
  for (let p = 0; p < ch.h * ch.w; p++) {
    const v = src[off + p] * 3;
    img.data.set([LUT[v], LUT[v + 1], LUT[v + 2], 255], p * 4);
  }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/** A small weight grid: blue for positive, red for negative, scaled by `max`. */
function kernelCanvas(k, max) {
  const h = k.length;
  const w = k[0].length;
  const canvas = document.createElement("canvas");
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const v = clamp(k[y][x] / max, -1, 1);
      const c = v >= 0 ? COLORS.blue : COLORS.red;
      const a = Math.abs(v);
      img.data.set([c[0] * a + 6 * (1 - a), c[1] * a + 8 * (1 - a), c[2] * a + 12 * (1 - a), 255], (y * w + x) * 4);
    }
  ctx.putImageData(img, 0, 0);
  return canvas;
}

/** Fill `box` with a clickable grid of channel maps; calls `onPick(c)` on selection. */
function mountChannelGrid(box, ch, sample, selectedChannel, onPick) {
  box.innerHTML = "";
  for (let c = 0; c < ch.c; c++) {
    const b = document.createElement("button");
    b.title = `Channel ${c}`;
    b.setAttribute("aria-label", `Channel ${c}`);
    b.setAttribute("aria-pressed", String(c === selectedChannel));
    b.appendChild(channelCanvas(ch, sample, c));
    b.addEventListener("click", () => onPick(c));
    box.appendChild(b);
  }
}

/** The channel whose map is brightest on average, a sensible first pick. */
function brightestChannel(ch, sample) {
  const src = bytes(ch.data[sample]);
  const n = ch.h * ch.w;
  let best = 0;
  let bestSum = -1;
  for (let c = 0; c < ch.c; c++) {
    let s = 0;
    for (let p = 0; p < n; p++) s += src[c * n + p];
    if (s > bestSum) [best, bestSum] = [c, s];
  }
  return best;
}

/**
 * Animate one output channel of a convolution: the kernel slides over the input and the
 * output map fills in, with the arithmetic for the current position underneath.
 * Returns a function that stops the animation.
 */
function mountConvolution(box, conv, node, sample, channel) {
  const inp = conv.in;
  const x = inp.data[sample];
  const K = conv.weight[channel];
  const kh = K[0].length;
  const kw = K[0][0].length;
  const [sh, sw] = conv.stride;
  const [ph, pw] = conv.padding;
  const H = inp.h;
  const W = inp.w;
  const Ho = Math.floor((H + 2 * ph - kh) / sh) + 1;
  const Wo = Math.floor((W + 2 * pw - kw) / sw) + 1;
  const bias = conv.bias ? conv.bias[channel] : 0;
  const act = conv.exact ? activationFn(conv.act) : null;
  const at = (c, y, xx) => (y < 0 || y >= H || xx < 0 || xx >= W ? 0 : x[(c * H + y) * W + xx]);

  // Precompute every output value.
  const z = new Float32Array(Ho * Wo);
  for (let oy = 0; oy < Ho; oy++)
    for (let ox = 0; ox < Wo; ox++) {
      let s = bias;
      for (let c = 0; c < inp.c; c++)
        for (let i = 0; i < kh; i++) for (let j = 0; j < kw; j++) s += K[c][i][j] * at(c, oy * sh - ph + i, ox * sw - pw + j);
      z[oy * Wo + ox] = s;
    }
  const a = act ? z.map(act) : z;
  let amax = 1e-9;
  for (const v of a) amax = Math.max(amax, Math.abs(v));

  // Input shown as the mean over its channels.
  const mean = new Float32Array(H * W);
  for (let c = 0; c < inp.c; c++) for (let p = 0; p < H * W; p++) mean[p] += x[c * H * W + p] / inp.c;
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of mean) [lo, hi] = [Math.min(lo, v), Math.max(hi, v)];
  const inCanvas = document.createElement("canvas");
  inCanvas.width = W;
  inCanvas.height = H;
  const inCtx = inCanvas.getContext("2d");
  const inImg = inCtx.createImageData(W, H);
  for (let p = 0; p < H * W; p++) {
    const g = Math.round(((mean[p] - lo) / (hi - lo || 1)) * 255);
    inImg.data.set([g, g, g, 255], p * 4);
  }
  inCtx.putImageData(inImg, 0, 0);
  const outCanvas = document.createElement("canvas");
  outCanvas.width = Wo;
  outCanvas.height = Ho;
  const outCtx = outCanvas.getContext("2d");
  const outImg = outCtx.createImageData(Wo, Ho);

  const kmax = Math.max(1e-9, ...K.flat(2).map(Math.abs));
  const actName = conv.act ? (conv.exact ? conv.act : `${node.tags.join(" → ")}`) : null;
  box.innerHTML = `
    <div class="muted">Kernel for channel ${channel}${inp.c > 1 ? `, one ${kh}×${kw} slice per input channel` : ""}</div>
    <div class="kgrid"></div>
    <canvas class="conv-anim" width="340" height="170" aria-label="Convolution animation"></canvas>
    <div class="row"><button class="btn" data-act="play">Pause</button><input type="range" min="0" max="${Ho * Wo - 1}" value="0" aria-label="Kernel position"></div>
    <div class="math"></div>`;
  const kgrid = box.querySelector(".kgrid");
  for (let c = 0; c < Math.min(inp.c, 32); c++) {
    const cell = document.createElement("div");
    cell.title = `Input channel ${c}`;
    cell.appendChild(kernelCanvas(K[c], kmax));
    kgrid.appendChild(cell);
  }
  const canvas = box.querySelector("canvas.conv-anim");
  const ctx = canvas.getContext("2d");
  const range = box.querySelector("input");
  const playBtn = box.querySelector('[data-act="play"]');
  const math = box.querySelector(".math");
  let pos = 0;
  let playing = true;
  let hold = 0;
  let last = performance.now();
  let raf = 0;
  playBtn.addEventListener("click", () => {
    playing = !playing;
    playBtn.textContent = playing ? "Pause" : "Play";
  });
  range.addEventListener("input", () => {
    playing = false;
    playBtn.textContent = "Play";
    pos = Number(range.value);
  });

  const pad = 8;
  const size = 150;
  function draw() {
    const p = Math.floor(pos);
    const oy = Math.floor(p / Wo);
    const ox = p % Wo;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.imageSmoothingEnabled = false;
    // Left: the input with the kernel window. Right: the output map so far.
    const s = size / Math.max(H, W);
    ctx.drawImage(inCanvas, pad, pad, W * s, H * s);
    ctx.strokeStyle = rgba(COLORS.accent, 1);
    ctx.lineWidth = 2;
    ctx.strokeRect(pad + (ox * sw - pw) * s, pad + (oy * sh - ph) * s, kw * s, kh * s);
    for (let q = 0; q < Ho * Wo; q++) {
      const v = q <= p ? Math.round((Math.abs(a[q]) / amax) * 255) * 3 : 0;
      outImg.data.set(q <= p ? [LUT[v], LUT[v + 1], LUT[v + 2], 255] : [10, 12, 16, 255], q * 4);
    }
    outCtx.putImageData(outImg, 0, 0);
    const so = size / Math.max(Ho, Wo);
    const x0 = canvas.width - pad - Wo * so;
    ctx.drawImage(outCanvas, x0, pad, Wo * so, Ho * so);
    ctx.strokeRect(x0 + ox * so, pad + oy * so, so, so);
    ctx.strokeStyle = "rgba(255,255,255,0.3)";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad, pad, W * s, H * s);
    ctx.strokeRect(x0, pad, Wo * so, Ho * so);
    ctx.fillStyle = "#8d939c";
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillText(inp.c > 1 ? `input (mean of ${inp.c})` : "input", pad, canvas.height - 6);
    ctx.fillText(`channel ${channel}`, x0, canvas.height - 6);
    range.value = String(p);

    // The arithmetic at this position.
    const zv = z[p];
    let html = "";
    if (inp.c === 1 && kh * kw <= 25) {
      html += `<table>`;
      for (let i = 0; i < kh; i++) {
        html += "<tr>";
        for (let j = 0; j < kw; j++) {
          const w = K[0][i][j];
          html += `<td class="num">${fmtNum(at(0, oy * sh - ph + i, ox * sw - pw + j))}</td><td class="w ${w < 0 ? "neg" : "pos"}">×${fmtNum(w)}</td>`;
        }
        html += "</tr>";
      }
      html += `</table>input × weight, summed`;
    } else {
      html += `Σ over ${inp.c} input channels × ${kh}×${kw} weights`;
    }
    html += ` ${bias < 0 ? "−" : "+"} bias ${fmtNum(Math.abs(bias))} = <b>${fmtNum(zv)}</b>`;
    if (act) html += `<br>${esc(conv.act)}(${fmtNum(zv)}) = <b>${fmtNum(a[p])}</b>`;
    else if (actName) html += `<br>then ${esc(actName)}`;
    html += `<br><span class="src">output position (${ox}, ${oy}) of ${Wo}×${Ho}</span>`;
    math.innerHTML = html;
  }
  function tick(now) {
    if (!canvas.isConnected) return;
    const dt = clamp((now - last) / 1000, 0, 0.1); // rAF time can precede performance.now()
    last = now;
    if (playing) {
      if (pos >= Ho * Wo - 1) {
        hold += dt;
        if (hold > 1.2) {
          pos = 0;
          hold = 0;
        }
      } else pos = Math.min(Ho * Wo - 1, pos + (dt * Ho * Wo) / 6); // whole map in ~6 s
    }
    draw();
    raf = requestAnimationFrame(tick);
  }
  draw();
  raf = requestAnimationFrame(tick);
  return () => cancelAnimationFrame(raf);
}
