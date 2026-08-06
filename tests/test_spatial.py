"""Tests for the Spatial Forensics Stream."""

import pytest
import torch
from streams.spatial.model import SpatialForensicsStream, CrossAttentionBlock


class TestCrossAttentionBlock:
    def test_forward_shape(self):
        block = CrossAttentionBlock(dim=256, num_heads=8)
        cnn_feat = torch.randn(2, 1, 256)
        vit_feat = torch.randn(2, 1, 256)
        output = block(cnn_feat, vit_feat)
        assert output.shape == (2, 1, 256)

    def test_different_batch_sizes(self):
        block = CrossAttentionBlock(dim=256, num_heads=8)
        for batch_size in [1, 4, 8]:
            cnn_feat = torch.randn(batch_size, 1, 256)
            vit_feat = torch.randn(batch_size, 1, 256)
            output = block(cnn_feat, vit_feat)
            assert output.shape == (batch_size, 1, 256)


class TestSpatialForensicsStream:
    def test_forward_pass(self, sample_tensor, device):
        model = SpatialForensicsStream(pretrained=False).to(device)
        model.eval()
        with torch.no_grad():
            result = model(sample_tensor.to(device))
        assert "spatial_score" in result
        assert result["spatial_score"].shape == (1,)
        assert 0.0 <= result["spatial_score"].item() <= 1.0

    def test_output_keys(self, sample_tensor, device):
        model = SpatialForensicsStream(pretrained=False).to(device)
        model.eval()
        with torch.no_grad():
            result = model(sample_tensor.to(device))
        expected_keys = {"spatial_score", "cnn_features", "vit_features", "fused_features"}
        assert set(result.keys()) == expected_keys

    def test_batch_processing(self, device):
        model = SpatialForensicsStream(pretrained=False).to(device)
        model.eval()
        batch = torch.randn(4, 3, 224, 224).to(device)
        with torch.no_grad():
            result = model(batch)
        assert result["spatial_score"].shape == (4,)
