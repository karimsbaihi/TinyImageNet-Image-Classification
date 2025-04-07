import torchvision.models as models
from torchvision.models import resnet18, ResNet18_Weights #for lightweight model.
import torch.nn as nn

def create_model(num_classes=200):
    # 1. Load pretrained ResNet‑34
    # model = models.resnet34(pretrained=True)
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)  # faster, lighter


    # 2. Adapt first conv layer (original expects 224×224 inputs)
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False
    )
    # 3. Remove the initial maxpool (keeps more spatial info)
    model.maxpool = nn.Identity()

    # 4. Replace the final fully‑connected layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
