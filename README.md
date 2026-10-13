# TorchScope

See what a PyTorch network looks like, and watch data flow through it, in the style of 3Blue1Brown's neural network videos.

```bash
pip install -e .
```

```python
import torchscope

viz = torchscope.visualize(model, images[:8], labels=class_names)
viz.save("model.html")  # open in a browser, or just display `viz` in Jupyter
```

The output is one self-contained HTML page:

- **Small MLPs** open in the **neuron view**: every neuron and weight (blue positive, red negative), the input image next to the input layer, and an animated forward pass where signals travel along the weights and neurons light up with their activations.
- **Everything else** (CNNs, ResNets, transformers) opens in the **3D layer view**: one slab per layer sized by its output shape and textured with its real activations, skip connections as arcs, and a pulse that travels through the network. Nested modules collapse into groups. Use the detail slider, double-click a group to open it, or click a bracket name to close it again.

Every sample in the batch can be picked in the viewer. Hover or click anything for details.

`torchscope.trace(model, x)` returns the underlying dataflow graph if you want to inspect it.

Examples (no downloads needed):

```bash
pip install -e ".[examples]"
python examples/digits.py              # trains a 784-16-16-10 MLP and a small CNN on synthetic digits
python examples/torchvision_models.py  # ResNet-18 and ViT-B/16
```

The 3D view loads three.js from a CDN, so it needs an internet connection; the neuron view works offline.
