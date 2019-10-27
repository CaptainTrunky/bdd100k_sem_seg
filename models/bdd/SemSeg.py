import torch as T
import torch.nn as nn

from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


class SemSeg(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.segm = deeplabv3_resnet101(pretrained=True)

        for param in self.segm.parameters():
            param.requires_grad = False

        self.segm.classifier = DeepLabHead(in_channels=2048, num_classes=num_classes)

    def forward(self, X):
        return self.segm(X)

    def get_losses(self):
        pass
