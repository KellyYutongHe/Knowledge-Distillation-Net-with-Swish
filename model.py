"""
Network achitecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import resnet18


class res_car(nn.Module):
    """
    Resnet for knee
    """
    def __init__(self, pretrained=True, num_classes=5):
        super().__init__()
        self.resnet = resnet18(pretrained=pretrained)
        # self.fc = nn.Linear(in_features=2048, out_features=512)
        self.output = nn.Linear(in_features=512, out_features=num_classes)

    def forward(self, x):
        x = self.resnet(x)
        out = self.output(x)

        return out
