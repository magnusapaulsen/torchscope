"""Train a tiny MLP and a tiny CNN on synthetic digits, then visualize both.

Run: pip install -e ".[examples]" && python examples/digits.py

The digits are rendered with matplotlib fonts (random size, rotation and offset), so
nothing is downloaded. The MLP is the 784-16-16-10 network from 3Blue1Brown's video
and opens in the neuron view; the CNN opens in the 3D layer view. Each also gets an
animated SVG that plays inside a GitHub README.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from torch import nn

import torchscope

OUT = Path(__file__).parent


def make_digits(per_class: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Render digits 0-9 as (N, 1, 28, 28) images in [0, 1] with labels (N,)."""
    rng = np.random.default_rng(seed)
    fig = Figure(figsize=(1, 1), dpi=56, facecolor="black")
    canvas = FigureCanvasAgg(fig)
    images, labels = [], []
    for digit in range(10):
        for _ in range(per_class):
            fig.clear()
            fig.text(
                0.5 + rng.uniform(-0.08, 0.08),
                0.5 + rng.uniform(-0.08, 0.08),
                str(digit),
                ha="center",
                va="center",
                color="white",
                fontsize=rng.uniform(30, 44),
                rotation=rng.uniform(-15, 15),
                family=rng.choice(["DejaVu Sans", "DejaVu Serif", "DejaVu Sans Mono"]),
                weight=rng.choice(["normal", "bold"]),
            )
            canvas.draw()
            images.append(np.asarray(canvas.buffer_rgba())[..., 0].copy())
            labels.append(digit)
    x = torch.tensor(np.stack(images), dtype=torch.float32)[:, None] / 255
    return F.avg_pool2d(x, 2), torch.tensor(labels)  # 56x56 -> 28x28


def train(model: nn.Module, x: torch.Tensor, y: torch.Tensor, epochs: int) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(epochs):
        for idx in torch.randperm(len(x)).split(128):
            opt.zero_grad()
            F.cross_entropy(model(x[idx]), y[idx]).backward()
            opt.step()


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    with torch.no_grad():
        return (model(x).argmax(1) == y).float().mean().item()


torch.manual_seed(0)
x_train, y_train = make_digits(per_class=300, seed=0)
x_test, y_test = make_digits(per_class=20, seed=1)
# One test image per digit, in a shuffled order, for the viewer's sample picker.
picks = torch.stack([torch.nonzero(y_test == d)[0, 0] for d in [3, 7, 0, 5, 2, 8, 1, 9]])

mlp = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 16), nn.ReLU(),
    nn.Linear(16, 16), nn.ReLU(),
    nn.Linear(16, 10),
)  # fmt: skip
train(mlp, x_train, y_train, epochs=15)
print(f"MLP test accuracy: {accuracy(mlp, x_test, y_test):.1%}")
viz = torchscope.visualize(mlp, x_test[picks], title="784-16-16-10 MLP on digits")
print(viz, "->", viz.save(OUT / "digits_mlp.html"))
print("README animation ->", viz.save_svg(OUT / "digits_mlp.svg"))

cnn = nn.Sequential(
    nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(16 * 7 * 7, 32), nn.ReLU(),
    nn.Linear(32, 10),
)  # fmt: skip
train(cnn, x_train, y_train, epochs=6)
print(f"CNN test accuracy: {accuracy(cnn, x_test, y_test):.1%}")
viz = torchscope.visualize(cnn, x_test[picks], title="Small CNN on digits")
print(viz, "->", viz.save(OUT / "digits_cnn.html"))
print("README animation ->", viz.save_svg(OUT / "digits_cnn.svg"))
