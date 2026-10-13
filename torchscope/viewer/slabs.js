// ---- Slab view: one 3D slab per layer, textured with activations, rendered with three.js ----

async function createSlabView(root, P, ui) {
  let THREE, OrbitControls, CSS2DRenderer, CSS2DObject;
  try {
    THREE = await import("three");
    ({ OrbitControls } = await import("three/addons/controls/OrbitControls.js"));
    ({ CSS2DRenderer, CSS2DObject } = await import("three/addons/renderers/CSS2DRenderer.js"));
  } catch (err) {
    const msg = document.createElement("div");
    msg.className = "notice";
    msg.textContent = "The 3D view loads three.js from cdn.jsdelivr.net, so it needs an internet connection.";
    root.appendChild(msg);
    return { duration: 1, frame() {}, setSample() {}, destroy: () => msg.remove() };
  }

  const nodes = P.graph.nodes;
  const consumers = nodes.map(() => []);
  nodes.forEach((n) => n.inputs.forEach((i) => consumers[i].push(n.id)));
  const maxDepth = Math.max(1, ...nodes.map((n) => n.scope.length));

  // ---- three.js setup ------------------------------------------------------------
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setClearColor(0x000000, 1);
  renderer.domElement.className = "view-canvas";
  root.appendChild(renderer.domElement);
  const labelRenderer = new CSS2DRenderer();
  labelRenderer.domElement.className = "labels";
  root.appendChild(labelRenderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 5000);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.screenSpacePanning = true;

  let sample = 0;
  let dirty = true;
  let lastT = -1;
  let follow = false;
  let hovered = null;
  let selected = null;
  let items = [];
  let edges = [];
  let pickables = [];
  let bandLabels = [];
  let world = new THREE.Group();
  scene.add(world);

  // ---- grouping by module scope ---------------------------------------------------
  let expanded = new Set();
  const prefix = (scope, k) => scope.slice(0, k).join(".");
  function setLevel(level) {
    expanded = new Set();
    for (const n of nodes) for (let k = 1; k < Math.min(level, n.scope.length); k++) expanded.add(prefix(n.scope, k));
  }
  function keyOf(n) {
    for (let k = 1; k < n.scope.length; k++) {
      const p = prefix(n.scope, k);
      if (!expanded.has(p)) return "g:" + p;
    }
    return "n:" + n.id;
  }
  function autoLevel() {
    for (let level = maxDepth; level >= 1; level--) {
      setLevel(level);
      if (new Set(nodes.map(keyOf)).size <= 48) return level;
    }
    return 1;
  }
  let level = autoLevel();
  setLevel(level);

  // ---- display graph and layout ----------------------------------------------------
  const SX = (d) => 0.3 + 0.42 * Math.log2(d + 1); // channels -> thickness along the flow
  const SY = (d) => 0.4 + 0.72 * Math.log2(d + 1); // spatial dims / tokens -> height, depth
  const GAP = 2.4;
  const STEP = 0.5; // seconds between ranks
  const FADE = 0.35;
  const T0 = 0.3;

  function buildDisplay() {
    const keys = nodes.map(keyOf);
    const map = new Map();
    nodes.forEach((n, i) => {
      if (!map.has(keys[i])) map.set(keys[i], { key: keys[i], group: keys[i].startsWith("g:"), members: [], preds: new Set(), succs: new Set() });
      map.get(keys[i]).members.push(n);
    });
    nodes.forEach((n, i) =>
      n.inputs.forEach((j) => {
        if (keys[j] !== keys[i]) {
          map.get(keys[j]).succs.add(keys[i]);
          map.get(keys[i]).preds.add(keys[j]);
        }
      }),
    );
    const list = [...map.values()];
    for (const it of list) {
      const ids = new Set(it.members.map((m) => m.id));
      const leaving = it.members.filter((m) => m.output || consumers[m.id].some((c) => !ids.has(c)));
      it.rep = (leaving.length ? leaving : it.members).at(-1);
      it.first = it.members[0].id;
      it.name = it.group ? it.key.slice(2) : it.rep.name;
      const parts = it.name.split(".");
      it.label = it.group ? (/^\d/.test(parts.at(-1)) && parts.length > 1 ? parts.slice(-2).join(".") : parts.at(-1)) : it.rep.label;
      it.params = it.members.reduce((s, m) => s + m.params, 0);
      it.glyph = !it.group && (it.rep.kind === "op" || it.rep.role === "reshape");
      const [c, h, w] = it.rep.dims || [1, 1, 1];
      it.size = it.glyph ? [0.9, 0.9, 0.9] : [SX(c), SY(h), SY(w)];
    }

    // Topological order (Kahn), ties broken by execution order.
    const indeg = new Map(list.map((it) => [it.key, it.preds.size]));
    const ready = list.filter((it) => it.preds.size === 0).sort((a, b) => a.first - b.first);
    const order = [];
    while (ready.length) {
      const it = ready.shift();
      order.push(it);
      for (const s of it.succs) {
        indeg.set(s, indeg.get(s) - 1);
        if (indeg.get(s) === 0) {
          ready.push(map.get(s));
          ready.sort((a, b) => a.first - b.first);
        }
      }
    }
    for (const it of list.sort((a, b) => a.first - b.first)) if (!order.includes(it)) order.push(it);
    order.forEach((it, i) => (it.index = i));

    for (const it of order) {
      it.rank = 0;
      for (const p of it.preds) if (map.get(p).index < it.index) it.rank = Math.max(it.rank, map.get(p).rank + 1);
    }
    for (const it of [...order].reverse()) {
      it.down = 0;
      for (const s of it.succs) if (map.get(s).index > it.index) it.down = Math.max(it.down, map.get(s).down + 1);
    }
    // The longest path runs straight along lane 0; branches get side lanes.
    let cur = order.filter((it) => it.preds.size === 0).sort((a, b) => b.down - a.down || a.first - b.first)[0];
    while (cur) {
      cur.main = true;
      cur = [...cur.succs].map((k) => map.get(k)).filter((s) => s.index > cur.index).sort((a, b) => b.down - a.down || a.first - b.first)[0];
    }
    const byRank = new Map();
    for (const it of order) {
      if (!byRank.has(it.rank)) byRank.set(it.rank, []);
      byRank.get(it.rank).push(it);
    }
    const laneGap = Math.max(...order.map((it) => it.size[2])) + 2.5;
    let x = 0;
    const ranks = [...byRank.keys()].sort((a, b) => a - b);
    const rankX = [];
    ranks.forEach((r, i) => {
      const row = byRank.get(r);
      const width = Math.max(...row.map((it) => it.size[0]));
      if (i > 0) x += width / 2;
      rankX[r] = x;
      let off = 0;
      for (const it of row.sort((a, b) => (b.main ? 1 : 0) - (a.main ? 1 : 0) || a.first - b.first)) {
        it.lane = it.main ? 0 : (off = off >= 0 ? -(off + 1) : -off); // -1, 1, -2, 2: branches go behind first
        it.x = x;
        it.z = it.lane * laneGap;
      }
      if (!row.some((it) => it.main) && row.length) {
        // No main item here: spread the row around the centre instead.
        row.forEach((it, j) => (it.z = (j - (row.length - 1) / 2) * laneGap));
      }
      x += width / 2 + GAP;
    });
    for (const it of order) it.t0 = T0 + it.rank * STEP;
    const maxRank = Math.max(0, ...order.map((it) => it.rank));
    return { order, map, rankX, maxRank };
  }

  // ---- textures --------------------------------------------------------------------
  const textures = new Map();
  function fillTexture(tex, node, face) {
    const d = tex.image.data;
    if (face === "image") {
      const img = node.image;
      const src = bytes(img.data[sample]);
      for (let y = 0; y < img.h; y++)
        for (let x = 0; x < img.w; x++) {
          const s = (y * img.w + x) * img.c;
          const o = ((img.h - 1 - y) * img.w + x) * 4;
          d[o] = src[s];
          d[o + 1] = src[img.c === 3 ? s + 1 : s];
          d[o + 2] = src[img.c === 3 ? s + 2 : s];
          d[o + 3] = 255;
        }
    } else {
      const f = node.faces[face];
      const src = bytes(f.data[sample]);
      for (let y = 0; y < f.h; y++)
        for (let x = 0; x < f.w; x++) {
          const v = src[y * f.w + x] * 3;
          const o = ((f.h - 1 - y) * f.w + x) * 4;
          d[o] = LUT[v];
          d[o + 1] = LUT[v + 1];
          d[o + 2] = LUT[v + 2];
          d[o + 3] = 255;
        }
    }
    tex.needsUpdate = true;
  }
  function texture(node, face) {
    const key = `${node.id}:${face}`;
    let tex = textures.get(key);
    if (!tex) {
      const f = face === "image" ? node.image : node.faces[face];
      tex = new THREE.DataTexture(new Uint8Array(f.w * f.h * 4), f.w, f.h, THREE.RGBAFormat);
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.magFilter = THREE.NearestFilter;
      tex.minFilter = THREE.LinearFilter;
      fillTexture(tex, node, face);
      textures.set(key, tex);
    }
    return tex;
  }

  // ---- scene objects ---------------------------------------------------------------
  const GLYPHS = { add: "+", sub: "−", mul: "×", div: "÷", matmul: "@", cat: "cat", stack: "stack" };
  const symbolOf = (n) => GLYPHS[n.type];
  const sphereGeo = new THREE.SphereGeometry(0.45, 28, 18);
  const particleGeo = new THREE.SphereGeometry(0.14, 10, 8);
  const particleMat = new THREE.MeshBasicMaterial({ color: 0x9fe8ff, transparent: true, opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false });
  const DIM_GLYPH = new THREE.Color(0x15232b);
  const LIT_GLYPH = new THREE.Color(0x58c4dd);
  const EDGE_DIM = new THREE.Color(0x1d3440);
  const EDGE_LIT = new THREE.Color(0x58c4dd);

  function makeItem(it) {
    const g = new THREE.Group();
    g.position.set(it.x, 0, it.z);
    const [sx, sy, sz] = it.size;
    if (it.glyph) {
      it.mats = [new THREE.MeshBasicMaterial({ color: DIM_GLYPH.clone() })];
      it.mesh = new THREE.Mesh(sphereGeo, it.mats[0]);
    } else {
      const n = it.rep;
      const front = n.image ? texture(n, "image") : texture(n, "front");
      const top = texture(n, "top");
      const side = texture(n, "side");
      it.mats = [front, front, top, top, side, side].map((map) => new THREE.MeshBasicMaterial({ map }));
      const box = new THREE.BoxGeometry(sx, sy, sz);
      it.mesh = new THREE.Mesh(box, it.mats);
      it.outline = new THREE.LineSegments(new THREE.EdgesGeometry(box), new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.3 }));
      g.add(it.outline);
      if (it.group) {
        const shell = new THREE.LineSegments(
          new THREE.EdgesGeometry(new THREE.BoxGeometry(sx + 0.5, sy + 0.5, sz + 0.5)),
          new THREE.LineDashedMaterial({ color: 0x58c4dd, dashSize: 0.25, gapSize: 0.18, transparent: true, opacity: 0.55 }),
        );
        shell.computeLineDistances();
        g.add(shell);
      }
    }
    it.mesh.userData.item = it;
    g.add(it.mesh);
    pickables.push(it.mesh);

    const symbol = it.glyph ? symbolOf(it.rep) : null;
    const div = document.createElement("div");
    div.className = "node-label" + (it.group ? " group" : "") + (symbol ? " glyph" : "");
    div.innerHTML = symbol ? esc(symbol) : `<span>${esc(it.label)}</span><small>${fmtShape(it.rep.shape)}</small>`;
    it.labelEl = div;
    it.labelObj = new CSS2DObject(div);
    it.labelObj.position.set(0, symbol ? 0 : -sy / 2 - 0.45, 0);
    it.labelObj.center.set(0.5, symbol ? 0.5 : 0);
    g.add(it.labelObj);
    it.obj = g;
    world.add(g);
  }

  function makeEdge(a, b, display) {
    let curve;
    if (b.rank - a.rank > 1 && a.lane === b.lane) {
      // Skip connection: arc over the slabs in between.
      let h = Math.max(a.size[1], b.size[1]) / 2;
      for (const it of display.order) if (it.lane === a.lane && it.rank > a.rank && it.rank < b.rank) h = Math.max(h, it.size[1] / 2);
      h += 1.4;
      const p0 = new THREE.Vector3(a.x, a.glyph ? 0.45 : a.size[1] / 2, a.z);
      const p3 = new THREE.Vector3(b.x, b.glyph ? 0.45 : b.size[1] / 2, b.z);
      curve = new THREE.CubicBezierCurve3(p0, new THREE.Vector3(a.x + 0.6, h + 0.6, a.z), new THREE.Vector3(b.x - 0.6, h + 0.6, b.z), p3);
    } else {
      const p0 = new THREE.Vector3(a.x + a.size[0] / 2, 0, a.z);
      const p3 = new THREE.Vector3(b.x - b.size[0] / 2, 0, b.z);
      const dx = (p3.x - p0.x) * 0.5;
      curve = new THREE.CubicBezierCurve3(p0, new THREE.Vector3(p0.x + dx, 0, p0.z), new THREE.Vector3(p3.x - dx, 0, p3.z), p3);
    }
    const mat = new THREE.LineBasicMaterial({ color: EDGE_DIM.clone(), transparent: true, opacity: 0.9 });
    const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(curve.getPoints(40)), mat);
    const particle = new THREE.Mesh(particleGeo, particleMat);
    particle.visible = false;
    world.add(line, particle);
    return { a, b, curve, line, mat, particle };
  }

  // Brackets under the slabs that show which expanded module each run of layers belongs to.
  // Clicking a bracket's name collapses that module again.
  function makeBands() {
    const bands = new Map();
    for (const it of items) {
      const comps = it.group ? it.name.split(".") : it.rep.scope;
      for (let k = 1; k < comps.length; k++) {
        const key = comps.slice(0, k).join(".");
        if (!bands.has(key)) bands.set(key, { key, depth: k, items: [] });
        bands.get(key).items.push(it);
      }
    }
    const list = [...bands.values()].filter((b) => b.items.length >= 2);
    if (!list.length) return;
    const deepest = Math.max(...list.map((b) => b.depth));
    const mat = new THREE.LineBasicMaterial({ color: 0x3f7f92, transparent: true, opacity: 0.9 });
    for (const b of list) {
      const x0 = Math.min(...b.items.map((it) => it.x - it.size[0] / 2));
      const x1 = Math.max(...b.items.map((it) => it.x + it.size[0] / 2));
      const z = Math.max(...b.items.map((it) => it.z + it.size[2] / 2));
      const half = Math.max(...b.items.map((it) => it.size[1] / 2));
      const y = -half - 1.9 - 0.9 * (deepest - b.depth);
      const pts = [[x0, y + 0.3], [x0, y], [x1, y], [x1, y + 0.3]].map(([x, yy]) => new THREE.Vector3(x, yy, z));
      world.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat));
      const div = document.createElement("button");
      div.className = "band-label";
      const parts = b.key.split(".");
      div.textContent = /^\d/.test(parts.at(-1)) && parts.length > 1 ? parts.slice(-2).join(".") : parts.at(-1);
      div.title = `${b.key}: click to collapse`;
      div.addEventListener("click", () => collapse(b.key));
      const obj = new CSS2DObject(div);
      obj.position.set(x0, y, z);
      obj.center.set(0, -0.15);
      world.add(obj);
      bandLabels.push(div);
    }
  }

  let display = null;
  let duration = 1;
  function disposeWorld() {
    world.traverse((o) => {
      if (o.geometry && o.geometry !== sphereGeo && o.geometry !== particleGeo) o.geometry.dispose();
      if (o.material && o.material !== particleMat) [].concat(o.material).forEach((m) => m.dispose());
    });
    for (const it of items) it.labelEl?.remove();
    for (const el of bandLabels) el.remove();
    bandLabels = [];
    scene.remove(world);
    world = new THREE.Group();
    scene.add(world);
  }
  function rebuild() {
    disposeWorld();
    pickables = [];
    hovered = null;
    display = buildDisplay();
    items = display.order;
    items.forEach(makeItem);
    edges = [];
    for (const it of items) for (const s of it.succs) edges.push(makeEdge(it, display.map.get(s), display));
    makeBands();
    duration = T0 + display.maxRank * STEP + 1.2;
    ui.setDuration(duration);
    ui.setStages(buildStages());
    if (selected) selected = items.find((it) => it.key === selected.key) || null;
    showPanel();
    dirty = true;
  }

  // One step per rank: what the layers that light up at that moment do.
  function buildStages() {
    const byRank = [];
    for (const it of items) (byRank[it.rank] ||= []).push(it);
    return byRank.map((row, r) => ({
      start: T0 + r * STEP,
      end: T0 + r * STEP + FADE,
      title: row.map((it) => it.label).join(" + "),
      text: () =>
        row
          .map((it) => {
            let t = esc(it.group ? groupExplain(it) : it.rep.explain || "");
            const top = it.rep.top?.[sample]?.[0];
            if (top) t += ` Prediction: <b>${esc(top[0])}</b> (${(top[1] * 100).toFixed(0)}%).`;
            return t;
          })
          .join(" "),
    }));
  }

  // ---- camera ----------------------------------------------------------------------
  const VIEW_DIR = new THREE.Vector3(-1, 0.55, 0.95).normalize();
  function frameItems(list) {
    const box = new THREE.Box3();
    for (const it of list) {
      const [sx, sy, sz] = it.size;
      box.expandByPoint(new THREE.Vector3(it.x - sx / 2, -sy / 2, it.z - sz / 2));
      box.expandByPoint(new THREE.Vector3(it.x + sx / 2, sy / 2 + 1, it.z + sz / 2));
    }
    const center = box.getCenter(new THREE.Vector3());
    const radius = box.getSize(new THREE.Vector3()).length() / 2;
    const fov = (camera.fov * Math.PI) / 180;
    const fit = Math.min(fov, fov * camera.aspect);
    const dist = Math.max(6, (radius / Math.sin(fit / 2)) * 1.15);
    controls.target.copy(center);
    camera.position.copy(center).addScaledVector(VIEW_DIR, dist);
    camera.near = dist / 200;
    camera.far = dist * 20 + 500;
    camera.updateProjectionMatrix();
    dirty = true;
  }
  function frameStart() {
    const length = items.length ? items.at(-1).x - items[0].x : 0;
    if (length < 70) {
      frameItems(items);
      setFollow(false);
    } else {
      frameItems(items.filter((it) => it.rank <= 5));
      setFollow(true);
    }
  }
  function pulseX(t) {
    const r = clamp((t - T0) / STEP, 0, display.maxRank);
    const r0 = Math.floor(r);
    const r1 = Math.min(display.maxRank, r0 + 1);
    const x0 = display.rankX[r0] ?? 0;
    const x1 = display.rankX[r1] ?? x0;
    return lerp(x0, x1, r - r0);
  }

  // ---- controls in the footer ------------------------------------------------------
  ui.legend.innerHTML =
    `<div><span class="ramp"></span>activation (low → high)</div>` +
    `<div class="muted">drag to orbit · scroll to zoom · double-click a group to expand</div>`;
  let levelInput = null;
  if (maxDepth > 1) {
    const label = document.createElement("label");
    label.innerHTML = `Detail <input type="range" min="1" max="${maxDepth}" step="1" value="${level}" aria-label="Detail level">`;
    levelInput = label.querySelector("input");
    levelInput.addEventListener("input", () => {
      level = Number(levelInput.value);
      setLevel(level);
      rebuild();
    });
    ui.extra.appendChild(label);
  }
  const followBtn = document.createElement("button");
  followBtn.className = "btn";
  followBtn.textContent = "Follow";
  followBtn.title = "Keep the camera on the pulse while playing";
  function setFollow(on) {
    follow = on;
    followBtn.setAttribute("aria-pressed", String(on));
  }
  followBtn.addEventListener("click", () => setFollow(!follow));
  const fitBtn = document.createElement("button");
  fitBtn.className = "btn";
  fitBtn.textContent = "Fit";
  fitBtn.title = "Show the whole model";
  fitBtn.addEventListener("click", () => {
    setFollow(false);
    frameItems(items);
  });
  ui.extra.append(followBtn, fitBtn);

  // ---- picking, tooltip and details panel -------------------------------------------
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let pointerEvent = null;
  let downAt = null;
  function pick(e) {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    const hit = raycaster.intersectObjects(pickables, false)[0];
    return hit ? hit.object.userData.item : null;
  }
  const typeLine = (n) => [n.type, ...n.tags].join(" + ");
  function onMove(e) {
    pointerEvent = e;
  }
  function updateHover() {
    if (!pointerEvent) return;
    const e = pointerEvent;
    pointerEvent = null;
    const it = pick(e);
    if (it !== hovered) {
      hovered = it;
      dirty = true;
      renderer.domElement.style.cursor = it ? "pointer" : "";
    }
    if (!it) return ui.hideTip();
    const n = it.rep;
    const what = it.group ? `${it.members.length} layers` : typeLine(n);
    ui.tip(`<b>${esc(it.label)}</b>${it.name !== it.label ? ` <span class="muted">${esc(it.name)}</span>` : ""}<br><span class="muted">${esc(what)}</span><br>${fmtShape(n.shape)}`, e.clientX, e.clientY);
  }
  let stopConv = null;
  let channel = null; // selected feature-map channel in the panel
  const groupExplain = (it) => {
    const kinds = [...new Set(it.members.filter((m) => m.kind === "module").map((m) => m.type))];
    return `${it.name} is a block of ${it.members.length} layers (${kinds.join(", ") || "operations"}). Double-click it, or press Expand, to look inside.`;
  };
  function showPanel() {
    stopConv?.();
    stopConv = null;
    if (!selected) return ui.panel(null);
    const it = selected;
    const n = it.rep;
    const sub = it.name !== it.label ? `<div class="muted">${esc(it.name)}</div>` : "";
    let html = `<div class="panel-head"><div><div class="panel-title">${esc(it.label)}</div>${sub}</div><button class="close" aria-label="Close">×</button></div>`;
    html += `<p class="explain">${esc(it.group ? groupExplain(it) : n.explain || "")}</p><dl>`;
    html += it.group ? `<dt>Contains</dt><dd>${it.members.length} layers</dd>` : `<dt>Type</dt><dd>${esc(typeLine(n))}</dd>`;
    html += `<dt>Output</dt><dd>${fmtShape(n.shape)}</dd>`;
    if (it.params) html += `<dt>Params</dt><dd>${fmtCount(it.params)}</dd>`;
    html += `</dl>`;
    if (!it.group && n.info) html += `<code>${esc(n.info)}</code>`;
    const actions = [];
    if (it.group) actions.push(`<button class="btn" data-act="expand">Expand</button>`);
    const parent = it.group ? it.name.split(".").slice(0, -1).join(".") : n.scope.slice(0, -1).join(".");
    if (parent) actions.push(`<button class="btn" data-act="collapse">Collapse into ${esc(parent)}</button>`);
    if (actions.length) html += `<div class="actions">${actions.join("")}</div>`;
    if (n.top) {
      html += `<h4>Top predictions</h4><table class="preds">`;
      for (const [label, p] of n.top[sample]) html += `<tr><td>${esc(label)}</td><td style="width:45%"><div class="bar" style="width:${(p * 100).toFixed(1)}%"></div></td><td style="text-align:right">${(p * 100).toFixed(1)}%</td></tr>`;
      html += `</table>`;
    }
    const conv = !it.group && n.conv;
    if (n.image) html += `<h4>Input image</h4><div data-canvas="image"></div>`;
    else if (n.channels) {
      const ch = n.channels;
      const count = ch.total > ch.c ? `first ${ch.c} of ${ch.total}` : `${ch.c}`;
      html += `<h4>Feature maps · ${count} channel${ch.total > 1 ? "s" : ""}</h4><div class="chgrid"></div>`;
      if (conv) html += `<h4>How a channel is computed</h4><div class="convbox"></div>`;
      else html += `<div data-canvas="channel"></div>`;
    } else if (n.faces) {
      const [c, h, w] = n.dims;
      if (h * w > 1) html += `<h4>Activation, mean over ${c > 1 ? c + " channels" : "channels"}</h4><div data-canvas="front"></div>`;
      if (c > 1) html += `<h4>Activation per channel</h4><div class="muted">rows: height, columns: channels</div><div data-canvas="side"></div>`;
    }
    const panel = ui.panel(html);
    panel.classList.toggle("wide", !!(n.channels || conv));
    for (const slot of panel.querySelectorAll("[data-canvas]")) {
      const kind = slot.dataset.canvas;
      if (kind === "image") slot.replaceWith(imageToCanvas(n.image, sample));
      else if (kind !== "channel") slot.replaceWith(faceToCanvas(n.faces[kind], sample));
    }
    if (n.channels && !n.image) {
      const ch = n.channels;
      if (channel == null || channel >= ch.c) channel = brightestChannel(ch, sample);
      const grid = panel.querySelector(".chgrid");
      const pick = (c) => {
        channel = c;
        mountChannelGrid(grid, ch, sample, c, pick);
        stopConv?.();
        const box = panel.querySelector(".convbox");
        if (box && conv) stopConv = mountConvolution(box, conv, n, sample, c);
        const slot = panel.querySelector('[data-canvas="channel"]');
        if (slot) {
          slot.innerHTML = `<div class="muted">Channel ${c}</div>`;
          slot.appendChild(channelCanvas(ch, sample, c));
        }
      };
      pick(channel);
    }
    panel.querySelector('[data-act="expand"]')?.addEventListener("click", () => expand(it));
    panel.querySelector('[data-act="collapse"]')?.addEventListener("click", () => collapse(parent));
    panel.querySelector(".close")?.addEventListener("click", () => {
      stopConv?.();
      selected = null;
      dirty = true;
    });
  }
  function expand(it) {
    expanded.add(it.name);
    selected = null;
    rebuild();
  }
  function collapse(name) {
    for (const k of [...expanded]) if (k === name || k.startsWith(name + ".")) expanded.delete(k);
    selected = { key: "g:" + name };
    rebuild();
  }
  const el = renderer.domElement;
  el.addEventListener("pointermove", onMove);
  el.addEventListener("pointerleave", () => {
    pointerEvent = null;
    hovered = null;
    dirty = true;
    ui.hideTip();
  });
  el.addEventListener("pointerdown", (e) => (downAt = [e.clientX, e.clientY]));
  el.addEventListener("pointerup", (e) => {
    if (!downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;
    const picked = pick(e);
    if (picked?.key !== selected?.key) channel = null;
    selected = picked;
    showPanel();
    dirty = true;
  });
  el.addEventListener("dblclick", (e) => {
    const it = pick(e);
    if (it && it.group) expand(it);
  });
  controls.addEventListener("change", () => (dirty = true));

  function resize() {
    const rect = root.getBoundingClientRect();
    renderer.setSize(rect.width, rect.height, false);
    renderer.domElement.style.width = rect.width + "px";
    renderer.domElement.style.height = rect.height + "px";
    labelRenderer.setSize(rect.width, rect.height);
    camera.aspect = rect.width / Math.max(1, rect.height);
    // Centre the picture in the free space between the header and the control bar.
    const shift = Math.max(0, ((document.getElementById("controls")?.offsetHeight || 0) - 60) / 2);
    camera.setViewOffset(rect.width, rect.height + 2 * shift, 0, 2 * shift, rect.width, rect.height);
    camera.updateProjectionMatrix();
    dirty = true;
  }
  const ro = new ResizeObserver(resize);
  ro.observe(root);
  ro.observe(document.getElementById("controls"));
  resize();
  rebuild();
  frameStart();

  // ---- animation -------------------------------------------------------------------
  const tmp = new THREE.Vector3();
  function update(t) {
    for (const it of items) {
      const a = smoothstep(it.t0, it.t0 + FADE, t);
      const flash = t >= it.t0 ? Math.exp(-(((t - it.t0 - FADE * 0.6) / 0.3) ** 2)) : 0;
      const isHot = it === hovered || it === selected;
      if (it.glyph) {
        it.mats[0].color.copy(DIM_GLYPH).lerp(LIT_GLYPH, a).multiplyScalar(1 + 0.6 * flash + (isHot ? 0.3 : 0));
        if (it.labelEl.classList.contains("glyph")) it.labelEl.classList.toggle("dim", a < 0.5);
      } else {
        const f = 0.1 + 0.9 * a + 0.55 * flash;
        for (const m of it.mats) m.color.setScalar(f);
        it.outline.material.opacity = isHot ? 1 : 0.18 + 0.35 * a;
        it.outline.material.color.set(isHot ? 0x58c4dd : 0xffffff);
      }
    }
    for (const e of edges) {
      const tA = e.a.t0 + FADE * 0.5;
      const tB = e.b.t0;
      const p = (t - tA) / Math.max(0.05, tB - tA);
      e.particle.visible = p > 0 && p < 1;
      if (e.particle.visible) e.particle.position.copy(e.curve.getPointAt(clamp(p, 0, 1)));
      const lit = smoothstep(0.9, 1.3, p);
      const hot = hovered && (e.a === hovered || e.b === hovered);
      e.mat.color.copy(EDGE_DIM).lerp(EDGE_LIT, hot ? 1 : lit * 0.75);
    }
    // Hide labels that are far from the camera, unless hovered or selected.
    const limit = 55;
    for (const it of items) {
      const d = camera.position.distanceTo(it.obj.getWorldPosition(tmp));
      it.labelObj.visible = d < limit + it.size[1] * 3 || it === hovered || it === selected || items.length < 14;
    }
  }

  return {
    get duration() {
      return duration;
    },
    get stages() {
      return buildStages();
    },
    frame(t) {
      updateHover();
      if (follow && ui.playing) {
        const dx = pulseX(t) - controls.target.x;
        const step = dx * 0.06;
        controls.target.x += step;
        camera.position.x += step;
        dirty = true;
      }
      if (controls.update()) dirty = true;
      if (t === lastT && !dirty) return;
      lastT = t;
      dirty = false;
      update(t);
      renderer.render(scene, camera);
      labelRenderer.render(scene, camera);
    },
    setSample(s) {
      sample = s;
      for (const [key, tex] of textures) {
        const [id, face] = key.split(":");
        fillTexture(tex, nodes[Number(id)], face);
      }
      showPanel();
      dirty = true;
    },
    destroy() {
      stopConv?.();
      ro.disconnect();
      controls.dispose();
      disposeWorld();
      for (const tex of textures.values()) tex.dispose();
      renderer.dispose();
      renderer.domElement.remove();
      labelRenderer.domElement.remove();
    },
  };
}
