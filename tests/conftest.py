"""Shared fixtures for SynthDoc test suite."""

import pytest
import torch
import numpy as np
from PIL import Image


@pytest.fixture
def sample_image():
    """Generate a synthetic test image (224x224 RGB)."""
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_tensor():
    """Generate a normalized image tensor for spatial stream."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def sample_freq_tensor():
    """Generate a 2-channel frequency tensor."""
    return torch.randn(1, 2, 224, 224)


@pytest.fixture
def device():
    """Return the appropriate test device."""
    return torch.device("cpu")
