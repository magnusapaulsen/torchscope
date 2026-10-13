"""Visualize ResNet-18 and ViT-B/16 from torchvision in the 3D layer view.

Run: pip install -e ".[examples]" && python examples/torchvision_models.py

weights=None keeps this offline, so activations come from random weights. Pass
weights="DEFAULT" (downloads the weights) and real images for meaningful activations,
and labels=weights.meta["categories"] for class names.
"""

from pathlib import Path

import torch
from torchvision.models import resnet18, vit_b_16

import torchscope

OUT = Path(__file__).parent
torch.manual_seed(0)
images = torch.rand(4, 3, 224, 224)

for name, model in [("resnet18", resnet18(weights=None)), ("vit_b_16", vit_b_16(weights=None))]:
    viz = torchscope.visualize(model, images, title=name)
    print(viz, "->", viz.save(OUT / f"{name}.html"))
