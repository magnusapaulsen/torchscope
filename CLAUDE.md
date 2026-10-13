# TorchScope
A PyTorch library that shows what a neural network looks like and animates data flowing through it, in the style of 3Blue1Brown's neural network videos. One call turns a model and a batch of example inputs into a self-contained interactive HTML page.

## Views
- **Neuron view** (small MLPs: a chain of `nn.Linear` layers). Every neuron is a node and every weight an edge: blue = positive, red = negative, thickness and opacity = |w|. Image inputs are drawn as a pixel grid next to the input column. The animation lights up each layer in turn as signals travel along the edges, weighted by `w * a`. Layers wider than `max_neurons` show their most active neurons.
- **Slab view** (everything else: CNNs, transformers, residual nets). One 3D slab per layer, sized by its output shape on a log scale, with faces textured by the real activations. Rotatable and zoomable (three.js). A pulse travels along the dataflow graph. Nested modules collapse into groups: a detail slider sets the depth, and double-clicking a group expands it.

## API
```python
import torchscope

# x is a batch of example inputs; each sample is selectable in the viewer.
viz = torchscope.visualize(model, x)
viz.save("net.html")  # or just display `viz` in Jupyter
viz.save_svg("net.svg")  # animated, plays in a GitHub README
graph = torchscope.trace(model, x)  # the simplified dataflow graph, for inspection and tests
```

## Commands
- Install: `pip install -e ".[dev]"` (the `examples` extra adds torchvision and matplotlib)
- Test: `pytest`
- Lint and format: `ruff check .` and `ruff format .`
- JS syntax check: `node --check torchscope/viewer/*.js`
- Preview examples: `python -m http.server 8765 --directory examples` (also in `.claude/launch.json`)

## Layout
- `torchscope/trace.py`: runs one forward pass and records module calls and tensor ops into a raw graph
- `torchscope/graph.py`: `Graph`/`Node` dataclasses and simplification
- `torchscope/summary.py`: turns activations into small per-sample summaries (slab face textures, vectors)
- `torchscope/export.py`: picks the view and builds the JSON payload
- `torchscope/explain.py`: plain-language captions for each layer (step mode, details panel)
- `torchscope/render.py`: `Visualization`; inlines the viewer assets and the JSON into one HTML file
- `torchscope/svg.py`, `torchscope/svg_slabs.py`: animated SVG export for READMEs (neuron view, slab view)
- `torchscope/viewer/`: `template.html`, `style.css`, and plain JS (`common.js`, `learn.js`, `neurons.js`, `slabs.js`, `main.js`), concatenated into one `<script type="module">`
- `tests/`: mirrors the package; toy models live in `tests/conftest.py`
- `examples/`: runnable scripts that write HTML files

## Tracing rules
- Trace with module forward hooks plus a `TorchFunctionMode`, not `torch.fx`, so any model with dynamic control flow works.
- Only tensors derived from the model inputs are tracked. Parameters and constants are not graph nodes.
- Leaf modules and `ATOMIC_TYPES` (e.g. `nn.MultiheadAttention`) become single nodes; ops inside them are not recorded. A non-leaf module whose forward created no nodes (e.g. `nn.TransformerEncoderLayer` on its fast path) also becomes a single node.
- Every node has a `scope`: the module path it was created in. The viewer uses it for grouping.
- Simplification: drop dropout/identity; fold activations, norms and shape-preserving unary ops into the preceding node when it has a single consumer (recorded in `tags`); merge chains of reshape-like ops.
- Keep tracked tensors alive for the whole trace so `id(tensor)` is never reused.

## Code rules
- Python 3.10+, type hints on public functions, Google-style docstrings (enforced by ruff).
- Never hardcode `.cuda()`. Use the input's device; move summaries to CPU.
- Do not leave the user's model changed. Restore `training` on every submodule, never touch `requires_grad` or `.grad`, and remove every hook and the function mode in `finally`.
- Core dependency is `torch` only (no numpy, no matplotlib).
- Keep the payload small: summaries are downsampled (faces at most 32 per axis, input images at most 64), at most `max_samples` samples, and bytes are base64-encoded.

## Viewer rules
- One self-contained HTML file, no build step, no npm. The only external dependency is three.js `0.160.0` from jsdelivr via an import map. Only the slab view imports it (dynamically), so the neuron view works offline.
- Dark 3Blue1Brown look. Colors are CSS tokens on `:root` in `style.css`: blue = positive, red = negative.
- Rendering is a pure function of the timeline position `t`, so the progress slider can scrub.

## SVG export rules
- The SVG must animate with CSS keyframes only: no `<script>`, no external references. GitHub renders README SVGs as images, so scripts never run.
- All animations share one looping duration (`_Timeline`), so they stay in sync. Images are embedded as PNG data URIs (`svg._png`, stdlib only).

## Audience
This is a learning tool, aimed first at MLPs and simple CNNs. Captions should explain what a layer does in plain language using its real sizes, and numbers shown to the user must be exact (or clearly labelled when they are not, as with folded BatchNorm).

## Testing
- Tests run on CPU, are seeded, and never download weights or data.
- Toy models in `tests/conftest.py` cover an MLP, a CNN, a residual CNN, a toy ViT with `nn.MultiheadAttention`, `nn.TransformerEncoderLayer`, and a module reused twice.
- Test the traced graph structure (nodes, edges, folding), the payload, view selection, and that the model is unchanged and hook-free afterwards.
- Viewer changes are checked in a browser: both views render without console errors, the animation plays and scrubs, sample switching works, and the detail slider works.

## Definition of done
A feature has tests, docstrings, and an example in `examples/`; `ruff`, `pytest` and `node --check` pass; and the viewer was checked in a browser.
