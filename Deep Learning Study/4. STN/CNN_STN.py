import numpy as np
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# AVG pooling: kernel=1or2, stride=2

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.Conv = nn.Sequential(
            # input: (-1, 1, 40, 40) if stn // (-1, 1, 80, 80) if no stn
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=9, stride=1, padding=0),
            # (-1, 32, 32, 32) // (-1, 32, 72, 72)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            # (-1, 32, 16, 16) // (-1, 32, 36, 36)
            
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=0), 
            # (-1, 64, 10, 10) // (-1, 64, 30, 30)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
            # (-1, 64, 5, 5) // (-1, 64, 15, 15)
            
            
        )
        
        
        self.classifier_STN = nn.Sequential(
            nn.Linear(64*5*5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
            
        )
        
        
        self.classifier_CNN = nn.Sequential(
            nn.Linear(64*15*15, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
            
        )
        
        
        self.localization = nn.Sequential(
            # input: (-1, 1, 80, 80)
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0), # (-1, 20, 76, 76)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), # (-1, 20, 38, 38)
            
            nn.Conv2d(in_channels=20, out_channels=20, kernel_size=5, stride=1, padding=0), # (-1, 20, 34, 34)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # (-1, 20, 17, 17)
            
        )
        
        self.fc_loc = nn.Sequential(
            nn.Linear(20*17*17, 20), 
            nn.ReLU(),
            nn.Linear(20, 6)
            
        )
        
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=np.float))
        
    def STN(self, x):
        y = self.localization(x)
        y = y.view(-1, 20*17*17)
        y = self.fc_loc(y) # (-1, 6)
        
        y = y.view(-1,2,3) # (-1, 2, 3)
        y = F.affine_grid(y, x.size()) # x.size(): (-1, 1, 80, 80) -> y: (-1, 80, 80, 2)
        out = F.grid_sample(x, y) # out: (-1, 1, 80, 80)
        
        out = self.avgpool(out) # out: (-1, 1, 40, 40)
        
        return out
        
        
        
    def forward(self, x):  
        # # below line for no use of STN
        x = self.STN(x)
        
        
        # conv
        y = self.Conv(x) 
        
        y = y.view(y.shape[0], -1)

        # # below line for no use of STN
        y = self.classifier_STN(y)
        
        # y = self.classifier_CNN(y)
        
        return y
        
            