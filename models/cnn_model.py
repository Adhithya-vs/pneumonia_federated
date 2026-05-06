import torch
import torch.nn as nn
from torchvision import models


class PneumoniaCNN(nn.Module):
    """
    Multi-label classifier for chest X-ray disease detection.
    Backbone : pretrained ResNet18 (ImageNet weights)
    Input    : grayscale images (1 channel), 224x224
    Output   : raw logits for [pneumonia, covid, tuberculosis]
    """
    def __init__(self, num_classes=3):
        super(PneumoniaCNN, self).__init__()

        # Load pretrained ResNet18
        self.base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Adapt first conv layer for 1-channel grayscale input
        self.base.conv1 = nn.Conv2d(
            1, 64,
            kernel_size=7, stride=2, padding=3, bias=False
        )

        # Replace final FC for multi-label output
        in_features = self.base.fc.in_features   # 512
        self.base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.base(x)

    # Convenience property used by GradCAM
    @property
    def conv3(self):
        return self.base.layer4[1].conv2