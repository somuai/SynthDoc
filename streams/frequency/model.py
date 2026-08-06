import torch
import torch.nn as nn
import torchvision.models as models


class FrequencyForensicsStream(nn.Module):
    """ResNet-18 based frequency domain forensic analyzer.
    Operates on concatenated FFT magnitude + DCT spectral maps (2-channel input)."""

    def __init__(self, pretrained=False):
        super().__init__()
        # Load standard ResNet-18
        base = models.resnet18(weights='DEFAULT' if pretrained else None)

        # Modify first conv layer to accept 2 channels instead of 3
        self.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # If pretrained, average the original 3-channel weights into 2 channels
        if pretrained:
            with torch.no_grad():
                orig_weight = base.conv1.weight
                # Average channels 0,1 for first channel and channel 2 for second
                self.conv1.weight[:, 0] = (orig_weight[:, 0] + orig_weight[:, 1]) / 2
                self.conv1.weight[:, 1] = orig_weight[:, 2]

        # Copy remaining ResNet layers
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        # Frequency-specific classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)  # (B, 512)
        score = self.classifier(features)  # (B, 1)

        return {
            'frequency_score': score.squeeze(-1),
            'frequency_confidence': torch.ones_like(score.squeeze(-1)) * 0.80,
            'frequency_features': features,
        }
