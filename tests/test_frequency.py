"""Tests for the Frequency Forensics Stream."""

import pytest
import torch
import numpy as np
from streams.frequency.model import FrequencyForensicsStream
from streams.frequency.preprocess import (
    compute_fft_magnitude, compute_dct_map, prepare_freq_tensor
)


class TestPreprocessing:
    def test_fft_magnitude_shape(self):
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = compute_fft_magnitude(gray)
        assert result.shape == (100, 100)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_dct_map_shape(self):
        gray = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        result = compute_dct_map(gray)
        assert result.shape == (100, 100)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_prepare_freq_tensor(self):
        rgb = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        tensor = prepare_freq_tensor(rgb, size=224)
        assert tensor.shape == (2, 224, 224)
        assert tensor.dtype == torch.float32


class TestFrequencyForensicsStream:
    def test_forward_pass(self, sample_freq_tensor, device):
        model = FrequencyForensicsStream(pretrained=False).to(device)
        model.eval()
        with torch.no_grad():
            result = model(sample_freq_tensor.to(device))
        assert "frequency_score" in result
        assert result["frequency_score"].shape == (1,)
        assert 0.0 <= result["frequency_score"].item() <= 1.0

    def test_output_keys(self, sample_freq_tensor, device):
        model = FrequencyForensicsStream(pretrained=False).to(device)
        model.eval()
        with torch.no_grad():
            result = model(sample_freq_tensor.to(device))
        assert "frequency_score" in result
        assert "frequency_features" in result
