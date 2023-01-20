import torch
import torch.nn as nn
import torch.nn.functional as F


class VGG_16(nn.Module):
        def __init__(self):
            super(VGG_16, self).__init__()
            
            self.Conv = nn.Sequential(
                # Conv1
                # input: -1x3x32x32
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),   # stride, padding -> fixed.
                nn.BatchNorm2d(64),
                nn.ReLU(),
                
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),    # 64x32x32
                nn.BatchNorm2d(64),
                nn.ReLU(),
                
                nn.MaxPool2d(kernel_size=2, stride=2),                                              # 64x16x16 //  fixed
                
         
                # Conv2
                # input: -1x64x16x16                                                                                     
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),    # 128x16x16
                nn.BatchNorm2d(128),
                nn.ReLU(),
                
                nn.MaxPool2d(kernel_size=2, stride=2),                                              # 128x8x8
         
                                                                                                  
                # Conv3
                # input: -1x128x8x8
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),    # 256x8x8
                nn.BatchNorm2d(256),
                nn.ReLU(),
                
                nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),    # 256x8x8
                nn.BatchNorm2d(256),
                nn.ReLU(),
                
                nn.MaxPool2d(kernel_size=2, stride=2),                                              # 256x4x4
                
                
                # Conv4
                # input: -1x256x4x4                                                                                  
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),    # 512x4x4
                nn.BatchNorm2d(512),
                nn.ReLU(),
                
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),    # 512x4x4
                nn.BatchNorm2d(512),
                nn.ReLU(),
                
                nn.MaxPool2d(kernel_size=2, stride=2),                                              # 512x2x2
                
                # Conv5
                # input: -1x512x2x2                                                                                  
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),    # 512x2x2
                nn.BatchNorm2d(512),
                nn.ReLU(),
                
                nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),    # 512x2x2
                nn.BatchNorm2d(512),
                nn.ReLU(),
                
                nn.MaxPool2d(kernel_size=2, stride=2)                                              # 512x1x1
                                                                                                   
            )
            
         
            
            self.FC = nn.Sequential(
                # FC1
                nn.Linear(512, 4096),   # 6*6*256 자리에 나중에 input shape이 변하면 -> 변수로 받는다.
                nn.ReLU(),
                
                # FC2
                nn.Linear(4096, 4096),
                nn.ReLU(),
                
                #FC3
                nn.Linear(4096, 10)
            )
        
        
        def forward(self, x):
            out = self.Conv(x)
            out = out.view(-1, 512)
            out = self.FC(out)
            
            return out