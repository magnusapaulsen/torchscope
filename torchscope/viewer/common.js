// ---- Shared helpers: payload decoding, colors, formatting, timeline and controls ----

const PAYLOAD = JSON.parse(document.getElementById("payload").textContent);

const COLORS = {
  blue: [35, 190, 240], // positive weights
  red: [250, 80, 40], // negative weights
  accent: [88, 196, 221],
};

const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const lerp = (a, b, t) => a + (b - a) * t;
const smoothstep = (a, b, x) => {
  const t = clamp((x - a) / (b - a), 0, 1);
  return t * t * (3 - 2 * t);
};
const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}
function fmtCount(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(1) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "k";
  return String(n);
}
function fmtShape(s) {
  return s && s.length ? s.join("×") : "scalar";
}
function fmtNum(v) {
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e4)) return v.toExponential(2);
  return v.toFixed(a < 1 ? 3 : 2);
}

const _bytesCache = new Map();
/** Decode a base64 string to a Uint8Array (cached). */
function bytes(b64) {
  let out = _bytesCache.get(b64);
  if (!out) {
    const bin = atob(b64);
    out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    _bytesCache.set(b64, out);
  }
  return out;
}

/** 256-entry RGB lookup table: dark navy -> blue -> cyan -> white. */
const LUT = (() => {
  const stops = [
    [0, [4, 8, 15]],
    [0.3, [16, 52, 92]],
    [0.6, [40, 140, 200]],
    [0.82, [110, 210, 240]],
    [1, [255, 255, 255]],
  ];
  const lut = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 0;
    while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
    const [t0, c0] = stops[k];
    const [t1, c1] = stops[k + 1];
    const u = (t - t0) / (t1 - t0);
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = Math.round(lerp(c0[c], c1[c], u));
  }
  return lut;
})();

/** Draw an `{c, h, w, data}` image (interleaved HWC bytes) for one sample into a canvas. */
function imageToCanvas(img, sample) {
  const canvas = document.createElement("canvas");
  canvas.width = img.w;
  canvas.height = img.h;
  const ctx = canvas.getContext("2d");
  const out = ctx.createImageData(img.w, img.h);
  const src = bytes(img.data[sample]);
  for (let p = 0; p < img.w * img.h; p++) {
    const o = p * 4;
    if (img.c === 3) {
      out.data[o] = src[p * 3];
      out.data[o + 1] = src[p * 3 + 1];
      out.data[o + 2] = src[p * 3 + 2];
    } else {
      out.data[o] = out.data[o + 1] = out.data[o + 2] = src[p];
    }
    out.data[o + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
  return canvas;
}

/** Draw a slab face `{h, w, data}` for one sample into a canvas, using the LUT. */
function faceToCanvas(face, sample) {
  const canvas = document.createElement("canvas");
  canvas.width = face.w;
  canvas.height = face.h;
  const ctx = canvas.getContext("2d");
  const out = ctx.createImageData(face.w, face.h);
  const src = bytes(face.data[sample]);
  for (let p = 0; p < face.w * face.h; p++) {
    const v = src[p] * 3;
    out.data[p * 4] = LUT[v];
    out.data[p * 4 + 1] = LUT[v + 1];
    out.data[p * 4 + 2] = LUT[v + 2];
    out.data[p * 4 + 3] = 255;
  }
  ctx.putImageData(out, 0, 0);
  return canvas;
}

/** Controls, timeline and the render loop. Views plug in via `ui.view`. */
function createUI(P) {
  const $ = (id) => document.getElementById(id);
  const ui = {
    stage: $("stage"),
    extra: $("extra"),
    legend: $("legend"),
    t: 0,
    duration: 1,
    playing: false,
    speed: 1,
    sample: 0,
    view: null,
    onSample: null,
  };

  $("title").textContent = P.title;
  const m = P.model;
  $("meta").textContent = `${m.name} · ${fmtCount(m.params)} parameters · input ${fmtShape(m.input_shape.slice(1))}`;
  document.title = P.title;

  const play = $("play");
  const icon = $("play-icon");
  const scrub = $("scrub");
  const ICONS = { play: "M8 5v14l11-7z", pause: "M6 5h4v14H6zM14 5h4v14h-4z", replay: "M12 5V1L7 6l5 5V7a5 5 0 1 1-5 5H5a7 7 0 1 0 7-7z" };

  function syncControls() {
    scrub.value = String(Math.round((ui.t / ui.duration) * 1000));
    const atEnd = ui.t >= ui.duration;
    icon.setAttribute("d", ui.playing ? ICONS.pause : atEnd ? ICONS.replay : ICONS.play);
    play.setAttribute("aria-label", ui.playing ? "Pause" : atEnd ? "Replay" : "Play");
  }
  ui.setDuration = (d) => {
    ui.duration = Math.max(0.1, d);
    ui.t = Math.min(ui.t, ui.duration);
    syncControls();
  };
  ui.restart = () => {
    ui.t = 0;
    ui.playing = true;
    syncControls();
  };
  ui.togglePlay = () => {
    if (ui.t >= ui.duration) ui.t = 0;
    ui.playing = !ui.playing;
    syncControls();
  };
  play.addEventListener("click", ui.togglePlay);
  scrub.addEventListener("input", () => {
    ui.playing = false;
    ui.t = (Number(scrub.value) / 1000) * ui.duration;
    syncControls();
  });
  $("speed").addEventListener("change", (e) => (ui.speed = Number(e.target.value)));
  window.addEventListener("keydown", (e) => {
    if (e.code === "Space" && !["INPUT", "SELECT", "BUTTON"].includes(document.activeElement?.tagName)) {
      e.preventDefault();
      ui.togglePlay();
    }
  });

  // Sample picker: thumbnails for image inputs, numbers otherwise.
  ui.buildSamples = (count, thumbs) => {
    const box = $("samples");
    box.innerHTML = "";
    if (count <= 1) return;
    for (let s = 0; s < count; s++) {
      const b = document.createElement("button");
      b.className = "sample";
      b.setAttribute("aria-label", `Sample ${s + 1}`);
      b.setAttribute("aria-pressed", String(s === ui.sample));
      if (thumbs && thumbs[s]) b.style.backgroundImage = `url(${thumbs[s]})`;
      else b.textContent = String(s + 1);
      b.addEventListener("click", () => {
        ui.sample = s;
        box.querySelectorAll(".sample").forEach((el, i) => el.setAttribute("aria-pressed", String(i === s)));
        ui.onSample?.(s);
      });
      box.appendChild(b);
    }
  };

  ui.buildViewSwitch = (kinds, current, onPick) => {
    const box = $("views");
    box.innerHTML = "";
    if (kinds.length < 2) return;
    const names = { neurons: "Neurons", slabs: "Layers 3D" };
    for (const k of kinds) {
      const b = document.createElement("button");
      b.textContent = names[k];
      b.setAttribute("role", "tab");
      b.setAttribute("aria-selected", String(k === current));
      b.addEventListener("click", () => {
        box.querySelectorAll("button").forEach((el) => el.setAttribute("aria-selected", String(el === b)));
        onPick(k);
      });
      box.appendChild(b);
    }
  };

  const tip = $("tooltip");
  ui.tip = (html, x, y) => {
    tip.innerHTML = html;
    tip.hidden = false;
    const r = tip.getBoundingClientRect();
    const left = x + 14 + r.width > window.innerWidth ? x - 14 - r.width : x + 14;
    const top = clamp(y + 14, 8, window.innerHeight - r.height - 8);
    tip.style.left = `${Math.max(8, left)}px`;
    tip.style.top = `${top}px`;
  };
  ui.hideTip = () => (tip.hidden = true);

  const panel = $("panel");
  ui.panel = (html) => {
    if (html == null) {
      panel.hidden = true;
      panel.innerHTML = "";
      return null;
    }
    panel.innerHTML = html;
    panel.hidden = false;
    panel.querySelector(".close")?.addEventListener("click", () => ui.panel(null));
    return panel;
  };

  ui.resetForView = () => {
    ui.extra.innerHTML = "";
    ui.legend.innerHTML = "";
    ui.panel(null);
    ui.hideTip();
  };

  let last = performance.now();
  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (ui.playing) {
      ui.t += dt * ui.speed;
      if (ui.t >= ui.duration) {
        ui.t = ui.duration;
        ui.playing = false;
      }
      syncControls();
    }
    ui.view?.frame(ui.t, dt);
    requestAnimationFrame(frame);
  }
  requestAnimationFrame(frame);
  syncControls();
  return ui;
}
