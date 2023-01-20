import torch
from torch import nn


class CAM(nn.Module):
    def __init__(self, features, num_classes, init_weights):
        super(CAM, self).__init__()
        
        self.features = features # VGG16 module에 해당
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))  # (1,1)크기로 stride=1만큼 이동 즉, 한 pixel씩 계산
        self.linear = nn.Linear(512, num_classes)
        
    def forward(self, x):
        y = self.features(x)
        map = y              # (-1, 512, 7, 7)
        y = self.avgpool(y)  # (-1, 512, 1)
        y = torch.squeeze(y) # (-1, 512)
        y = self.linear(y)   # (-1, 64)
        
        return y, map
        
        