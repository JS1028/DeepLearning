import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, activation=True, **kwargs):
        super(Conv_block, self).__init__()
        
        self.ReLU = nn.ReLU()
        self.Conv = nn.Conv2d(in_channels, out_channels, **kwargs) # **kwargs 비추
        self.BN = nn.BatchNorm2d(out_channels)
        self.activation = activation
        
    def forward(self, x):
        if not self.activation:
            return self.BN(self.Conv(x))
        return self.ReLU(self.BN(self.Conv(x))) # return에 여러 함수x
    
    
    

class Res_block(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(Res_block, self).__init__()
        
        
        if in_channels == 64:
            self.conv_seq = nn.Sequential(         
                Conv_block(in_channels, mid_channels, kernel_size=1),  #stride=1(default)
                Conv_block(mid_channels, mid_channels, kernel_size=3, padding=1),
                Conv_block(mid_channels, out_channels, activation=False, kernel_size=1))
            self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1) # depth만 바꿔주는 identity
                
        elif in_channels == out_channels:
            self.conv_seq = nn.Sequential(
                Conv_block(in_channels, mid_channels, kernel_size=1),  
                Conv_block(mid_channels, mid_channels, kernel_size=3, padding=1),
                Conv_block(mid_channels, out_channels, activation=False, kernel_size=1))
            self.identity = nn.Identity()
        
        # Down-sampling
        else:  
            self.conv_seq = nn.Sequential(
                Conv_block(in_channels, mid_channels, kernel_size=1, stride=2), 
                Conv_block(mid_channels, mid_channels, kernel_size=3, padding=1),
                Conv_block(mid_channels, out_channels, activation=False, kernel_size=1))
            self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2)  #padding=0(default)
        
        self.ReLU = nn.ReLU()
        
    def forward(self, x):
        y = self.conv_seq(x)
        y += self.identity(x)
        y = self.ReLU(y)
        
        return y
        
        
        
        
                              
        

class ResNet_50(nn.Module):
        def __init__(self): 
            super(ResNet_50, self).__init__()
            
            self.Conv1 = Conv_block(3, 64, kernel_size=3, padding=1)
                # 64x32x32
                
            self.Conv2 = nn.Sequential(
                Res_block(64, 64, 256),
                Res_block(256, 64, 256),
                Res_block(256, 64, 256)
                
            )
                # 256x32x32
                
            self.Conv3 = nn.Sequential(
                Res_block(256, 128, 512),
                Res_block(512, 128, 512),
                Res_block(512, 128, 512),
                Res_block(512, 128, 512)
            )
                # 512x16x16
                
            self.Conv4 = nn.Sequential(
                Res_block(512, 256, 1024),
                Res_block(1024, 256, 1024),
                Res_block(1024, 256, 1024),
                Res_block(1024, 256, 1024),
                Res_block(1024, 256, 1024),
                Res_block(1024, 256, 1024)
            )
                # 1024x8x8
            
            self.Conv5 = nn.Sequential(
                Res_block(1024, 512, 2048),
                Res_block(2048, 512, 2048),
                Res_block(2048, 512, 2048)
                
            )
                # 2048x4x4
                
            self.avg_pool = nn.AvgPool2d(kernel_size=4, stride=1)
            self.FC = nn.Linear(2048, 10)
            
        def forward(self, x):
            x = self.Conv1(x)
            x = self.Conv2(x)
            x = self.Conv3(x)
            x = self.Conv4(x)
            x = self.Conv5(x)
            x = self.avg_pool(x)
            x = x.view(x.shape[0], -1)
            x = self.FC(x)
            
            return x
                
                
                                                                                                   
        
        
 