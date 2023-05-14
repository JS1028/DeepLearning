import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class VGG19Gray(nn.Module):
    def __init__(self):
        super(VGG19Gray, self).__init__()
        vgg16 = models.vgg19(pretrained=True)
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            *list(vgg16.features.children())[1:]
        )

    def forward(self, x):
        x = self.features(x)
        return x