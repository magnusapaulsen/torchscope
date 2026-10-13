import pytest
import torch

from torchscope.summary import FACE_MAX, IMAGE_MAX, canonical, summarize


@pytest.mark.parametrize(
    "shape, dims",
    [
        ((2,), (1, 1, 1)),
        ((2, 10), (1, 10, 1)),
        ((2, 7, 16), (16, 7, 1)),
        ((2, 3, 5, 6), (3, 5, 6)),
        ((2, 2, 3, 5, 6), (6, 5, 6)),
    ],
)
def test_canonical_dims(shape, dims):
    t = torch.randn(shape)
    assert summarize(t).dims == dims
    assert canonical(t.reshape(shape) if len(shape) > 1 else t.reshape(2)).shape[1:] == dims


def test_faces_are_downsampled():
    s = summarize(torch.randn(2, 100, 64, 48))
    assert s.front.shape == (2, FACE_MAX, FACE_MAX)
    assert s.side.shape == (2, FACE_MAX, FACE_MAX)  # (h, c)
    assert s.top.shape == (2, FACE_MAX, FACE_MAX)  # (w, c)
    small = summarize(torch.randn(2, 4, 5, 6))
    assert small.front.shape == (2, 5, 6)
    assert small.side.shape == (2, 5, 4)
    assert small.top.shape == (2, 6, 4)


def test_faces_are_means():
    t = torch.randn(2, 4, 5, 6)
    s = summarize(t)
    assert torch.allclose(s.front, t.mean(1))
    assert torch.allclose(s.side, t.mean(3).transpose(1, 2))
    assert torch.allclose(s.top, t.mean(2).transpose(1, 2))


def test_vector_kept_only_when_small():
    assert summarize(torch.randn(2, 10)).vector.shape == (2, 10)
    assert summarize(torch.randn(1, 3, 128, 128)).vector is None


def test_image_only_for_image_like_inputs():
    assert summarize(torch.rand(2, 3, 200, 100), image=True).image.shape == (2, 3, IMAGE_MAX, 32)
    assert summarize(torch.rand(2, 28, 28), image=True).image.shape == (2, 1, 28, 28)
    assert summarize(torch.rand(2, 8, 28, 28), image=True).image is None
    assert summarize(torch.rand(2, 3, 28, 28)).image is None


def test_integer_and_bool_tensors():
    assert summarize(torch.randint(0, 9, (2, 7))).dims == (1, 7, 1)
    assert summarize(torch.ones(2, 3, dtype=torch.bool)).vector.dtype == torch.float32
