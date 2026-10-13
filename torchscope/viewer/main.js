// ---- Entry point: build the controls and open the initial view ----

(function main() {
  const P = PAYLOAD;
  const ui = createUI(P);

  // Sample thumbnails come from the input image, if the input looks like one.
  const inputImage = P.neurons?.image || P.graph?.nodes?.[0]?.image;
  const thumbs = inputImage ? Array.from({ length: P.samples }, (_, s) => imageToCanvas(inputImage, s).toDataURL()) : null;
  ui.buildSamples(P.samples, thumbs);

  let current = null;
  let token = 0;
  async function show(kind) {
    const mine = ++token;
    if (current) current.destroy();
    current = null;
    ui.view = null;
    ui.resetForView();
    const view = kind === "neurons" ? createNeuronView(ui.stage, P, ui) : await createSlabView(ui.stage, P, ui);
    if (mine !== token) return view.destroy();
    view.setSample(ui.sample);
    current = view;
    ui.view = view;
    ui.setDuration(view.duration);
    ui.setStages(view.stages || []);
    ui.restart();
  }
  ui.onSample = (s) => {
    current?.setSample(s);
    ui.restart();
  };
  ui.buildViewSwitch(P.neurons ? ["neurons", "slabs"] : ["slabs"], P.view, show);
  show(P.view);
})();
