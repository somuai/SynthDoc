import torch
import torch.nn as nn
import timm


class CrossAttentionBlock(nn.Module):
    """Cross-attention mechanism to fuse CNN local features with ViT global features."""
    def __init__(self, dim=256, num_heads=8):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, cnn_features, vit_features):
        q = self.query(cnn_features)
        k = self.key(vit_features)
        v = self.value(vit_features)
        attn_out, _ = self.attn(q, k, v)
        x = self.norm(attn_out + cnn_features)
        x = x + self.ffn(x)
        return x


class SpatialForensicsStream(nn.Module):
    """Hybrid EfficientNet-B4 + ViT architecture with cross-attention fusion
    for detecting pixel-level manipulation artifacts."""

    def __init__(self, pretrained=True):
        super().__init__()
        # CNN backbone: EfficientNet-B4
        self.cnn = timm.create_model('efficientnet_b4', pretrained=pretrained, num_classes=0)
        cnn_dim = self.cnn.num_features  # 1792 for efficientnet_b4

        # ViT backbone
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=pretrained, num_classes=0)
        vit_dim = self.vit.num_features  # 384 for vit_small

        # Project both to common dimension
        self.cnn_proj = nn.Linear(cnn_dim, 256)
        self.vit_proj = nn.Linear(vit_dim, 256)

        # Cross-attention fusion
        self.cross_attn = CrossAttentionBlock(dim=256, num_heads=8)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Extract features from both backbones
        cnn_feat = self.cnn(x)              # (B, 1792)
        vit_feat = self.vit(x)              # (B, 384)

        # Project to common space
        cnn_proj = self.cnn_proj(cnn_feat).unsqueeze(1)   # (B, 1, 256)
        vit_proj = self.vit_proj(vit_feat).unsqueeze(1)   # (B, 1, 256)

        # Cross-attention fusion
        fused = self.cross_attn(cnn_proj, vit_proj)       # (B, 1, 256)
        fused = fused.squeeze(1)                           # (B, 256)

        # Final prediction
        score = self.classifier(fused)      # (B, 1)

        return {
            'spatial_score': score.squeeze(-1),
            'cnn_features': cnn_feat,
            'vit_features': vit_feat,
            'fused_features': fused,
        }
