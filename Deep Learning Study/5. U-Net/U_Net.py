import numpy as np
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=(3,3), stride=1, padding=1),
            # (-1, out_channels, n, m)
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            # (-1, out_channels, n, m)
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
            
            
        )
        
        self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        
    def forward(self, x):
        

def stack_encoder(in_channels, out_channels):
    # input: (-1, in_channels, n, m) 
    out = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1),
        # (-1, out_channels, n, m)
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        # (-1, out_channels, n, m)
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()


    )
    return out

def stack_decoder(in_channels, out_channels):
    # input: (-1, in_channels, n, m) 
    out = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1),
        # (-1, out_channels, n, m)
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        # (-1, out_channels, n, m)
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        
        nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3), stride=1, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()


    )
    return out

def maxpool():
    # (-1, out_channels, n, m)
    out = nn.MaxPool2d(kernel_size=(2,2), stride=2) 
    # (-1, out_channels, n/2, m/2)
    return out

def upsampling(in_channels, out_channels, out_pad):
     out = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(2,2), stride=2, output_padding=out_pad)
     
     return out

def downsampling(in_channels, out_channels):
    out = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1), stride=1, padding=0)
    
    return out
    
def concat(from_decod, from_encod):
    result = torch.cat((from_decod, from_encod), dim=1)
    return result
    
            
class U_Net(nn.Module):
    def __init__(self):
        super(U_Net, self).__init__()
        
        self.enc_repeat_1 = stack_encoder(3, 24)
        self.enc_repeat_2 = stack_encoder(24, 64)
        self.enc_repeat_3 = stack_encoder(64, 128)
        self.enc_repeat_4 = stack_encoder(128, 256)
        self.enc_repeat_5 = stack_encoder(256, 512)
            
        self.enc_maxpool = maxpool()
        
        self.bridge = stack_encoder(512, 512)

       
        
        self.dec_upsampling_1 = upsampling(512, 512, (1,1))
        #self.enc_downsampling_1 = downsampling(512, 256)
        self.dec_repeat_1 = stack_decoder(1024, 256)
        
        self.dec_upsampling_2 = upsampling(256, 256, (0,1))
        #self.enc_downsampling_2 = downsampling(256, 128)
        self.dec_repeat_2 = stack_decoder(512, 128)
        
        self.dec_upsampling_3 = upsampling(128, 128, (1,0))
        #self.enc_downsampling_3 = downsampling(128, 64)
        self.dec_repeat_3 = stack_decoder(256, 64)
        
        self.dec_upsampling_4 = upsampling(64, 64, (0,1))
        #self.enc_downsampling_4 = downsampling(64, 32)
        self.dec_repeat_4 = stack_decoder(128, 24)
        
        self.dec_upsampling_5 = upsampling(24, 24, (0,0))
        #self.enc_downsampling_5 = downsampling(24, 12)
        self.dec_repeat_5 = stack_decoder(48, 24)
        
        self.last_conv = downsampling(24, 3)
        
    def forward(self, x):  
        
        enc_y1 = self.enc_repeat_1(x)
        y1 = self.enc_maxpool(enc_y1)
        
        
        enc_y2 = self.enc_repeat_2(y1)
        y2 = self.enc_maxpool(enc_y2)
        
        
        enc_y3 = self.enc_repeat_3(y2)
        y3 = self.enc_maxpool(enc_y3)
    
        
        enc_y4 = self.enc_repeat_4(y3)
        y4 = self.enc_maxpool(enc_y4)
      
        enc_y5 = self.enc_repeat_5(y4)
        y5 = self.enc_maxpool(enc_y5)
        
        bridge = self.bridge(y5)
        
        
        dec_y5 = self.dec_upsampling_1(bridge)
        y5 = concat(dec_y5, enc_y5)
        y5 = self.dec_repeat_1(y5)
        
        dec_y4 = self.dec_upsampling_2(y5)
        y4 = concat(dec_y4, enc_y4)
        y4 = self.dec_repeat_2(y4)
        
        dec_y3 = self.dec_upsampling_3(y4)
        y3 = concat(dec_y3, enc_y3)
        y3 = self.dec_repeat_3(y3)
        
        dec_y2 = self.dec_upsampling_4(y3)
        y2 = concat(dec_y2, enc_y2)
        y2 = self.dec_repeat_4(y2)
        
        dec_y1 = self.dec_upsampling_5(y2)
        
        y1 = concat(dec_y1, enc_y1)
        y1 = self.dec_repeat_5(y1)
        
        y = self.last_conv(y1)
        
        
        
        
        
        
        return y
        



    
