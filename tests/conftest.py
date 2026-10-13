import pytest
import torch
import torch.nn.functional as F
from torch import nn


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def mlp():
    return nn.Sequential(
        nn.Flatten(), nn.Linear(64, 12), nn.ReLU(), nn.Linear(12, 8), nn.ReLU(), nn.Linear(8, 4)
    )


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.drop = nn.Dropout(0.5)
        self.fc = nn.Linear(8 * 2 * 2, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.drop(F.relu(self.conv2(x)))
        return self.fc(torch.flatten(self.pool(x), 1))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.relu = nn.ReLU()  # called twice, like torchvision's BasicBlock

    def forward(self, x):
        out = self.relu(self.conv1(x))
        return self.relu(self.conv2(out) + x)


class ResNetish(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 4, 3, padding=1)
        self.layer1 = nn.Sequential(Block(4), Block(4))
        self.head = nn.Linear(4, 3)

    def forward(self, x):
        x = self.layer1(self.stem(x))
        return self.head(x.mean(dim=(2, 3)))


class ViTBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, 2, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class TinyViT(nn.Module):
    """16x16 input, 4x4 patches -> 1 CLS + 16 patch tokens of dim 8."""

    def __init__(self, d=8):
        super().__init__()
        self.patch = nn.Conv2d(3, d, 4, stride=4)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, 17, d))
        self.blocks = nn.Sequential(ViTBlock(d), ViTBlock(d))
        self.head = nn.Linear(d, 5)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1) + self.pos
        return self.head(self.blocks(x)[:, 0])


def encoder():
    layer = nn.TransformerEncoderLayer(8, 2, 16, batch_first=True)
    return nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)


# name -> (model factory, input shape)
MODELS = {
    "mlp": (mlp, (3, 1, 8, 8)),
    "cnn": (TinyCNN, (3, 1, 8, 8)),
    "resnet": (ResNetish, (3, 3, 8, 8)),
    "vit": (TinyViT, (3, 3, 16, 16)),
    "encoder": (encoder, (3, 5, 8)),
}


@pytest.fixture(params=list(MODELS))
def case(request):
    factory, shape = MODELS[request.param]
    return request.param, factory(), torch.randn(shape)


def snapshot(model):
    return {
        "training": {n: m.training for n, m in model.named_modules()},
        "requires_grad": {n: p.requires_grad for n, p in model.named_parameters()},
        "grads": {n: p.grad for n, p in model.named_parameters()},
        "params": {n: p.detach().clone() for n, p in model.named_parameters()},
        "hooks": {
            n: (len(m._forward_hooks), len(m._forward_pre_hooks)) for n, m in model.named_modules()
        },
    }


def assert_unchanged(model, before):
    after = snapshot(model)
    for key in ("training", "requires_grad", "grads", "hooks"):
        assert after[key] == before[key], key
    for n, p in before["params"].items():
        assert torch.equal(after["params"][n], p), n
    assert torch.overrides._get_current_function_mode_stack() == []
