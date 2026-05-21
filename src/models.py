import torch.nn as nn
from torchvision import models


def get_resnet18(num_classes: int, freeze_backbone: bool = True):
    """
    Loads a pretrained ResNet18 and replaces the final layer.
    """

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model