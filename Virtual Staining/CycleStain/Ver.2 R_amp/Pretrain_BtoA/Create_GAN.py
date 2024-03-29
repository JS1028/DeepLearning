import numpy as np
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os


class block_A(nn.Module):
    def __init__(self, i_c, o_c, dec=False):
        super(block_A, self).__init__()
        self.i_c = i_c
        self.o_c = o_c
        self.dec = dec
        
        if self.dec == False:
            func = nn.InstanceNorm2d(self.o_c)
        else:
            func = nn.Tanh()
        
        self.main = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=self.i_c, out_channels=self.o_c, kernel_size=7, stride=1, padding=0),
            func
        )
        
      
        
    def forward(self, x):
        # print("_c7s1:", self.main(x).size())
        
        
                
                
        
        return self.main(x)

            
            
class block_B(nn.Module):
    def __init__(self, i_c, o_c, dec=False):
        super(block_B, self).__init__()
        self.i_c = i_c
        self.o_c = o_c
        self.dec = dec
        
        if self.dec == False:
            conv = nn.Conv2d(in_channels=self.i_c, out_channels=self.o_c, kernel_size=3, stride=2, padding=1)
        else:
            conv = nn.ConvTranspose2d(in_channels=self.i_c, out_channels=self.o_c, kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.main= nn.Sequential(
            conv,
            nn.InstanceNorm2d(self.o_c),
            nn.ReLU(inplace=False)
                         
        )
        

        
    def forward(self, x):
        # print("c7s1:", self.main(x).size())
        
        
        return self.main(x)

    
    
    
class Res_block(nn.Module):
    def __init__(self):
        super(Res_block, self).__init__()
        
        self.res = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=False),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.InstanceNorm2d(256)

        )
#         for layer in self.res.modules():
#             if isinstance(layer, nn.ConvTranspose2d):
#                 layer.weight.data.normal_(0.0, 0.02)
#                 print('Gen_res_ConvTranspose 성공')
#             elif isinstance(layer, nn.Conv2d):
#                 layer.weight.data.normal_(0.0, 0.02)
#                 print('Gen_res_Conv2d 성공')
    
    def forward(self, x):
        # print("res_block:", (x+self.res(x)).size())
        return x + self.res(x)
    

class Dis_block(nn.Module):
    def __init__(self, i_c, o_c, stride=2, first=False, last=False):
        super(Dis_block, self).__init__()
        
        self.i_c = i_c
        self.o_c = o_c
        self.stride = stride
        self.first = first
        self.last = last
        
        if self.first == True:
            self.main = nn.Sequential(
                nn.Conv2d(in_channels=self.i_c, out_channels=self.o_c, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=False)
                
            )
        
        elif self.last == False:
            self.main = nn.Sequential(
                nn.Conv2d(in_channels=self.i_c, out_channels=self.o_c, kernel_size=4, stride=self.stride, padding=1),
                nn.InstanceNorm2d(self.o_c),
                nn.LeakyReLU(0.2, inplace=False)
                
            )
    
        else:
            # self.main = nn.Conv2d(in_channels=self.i_c, out_channels=self.o_c, kernel_size=16, stride=1, padding=0)
            # For 300x300
#             self.main = nn.Conv2d(in_channels=self.i_c, out_channels=self.o_c, kernel_size=18, stride=1, padding=0)
            
            # 600x600
#             self.main = nn.Conv2d(in_channels=self.i_c, out_channels=self.o_c, kernel_size=37, stride=1, padding=0)
            # 512x512
#             self.main = nn.Conv2d(in_channels=self.i_c, out_channels=self.o_c, kernel_size=32, stride=1, padding=0)
            # 256x256
            self.main = nn.Conv2d(in_channels=self.i_c, out_channels=self.o_c, kernel_size=16, stride=1, padding=0)
        
#         for layer in self.main.modules():
#             if isinstance(layer, nn.ConvTranspose2d):
#                 layer.weight.data.normal_(0.0, 0.02)
#                 print('Dis_block_ConvTranspose 성공')
#             elif isinstance(layer, nn.Conv2d):
#                 layer.weight.data.normal_(0.0, 0.02)
#                 print('Dis_block_Conv2d 성공')
    
    
    def forward(self, x):
        # print("dis_block:", self.main(x).size())
        return self.main(x)
    
######################################################################################3   
    
    
class Generator(nn.Module):
    def __init__(self, res_num, first_ch, last_ch):
        super(Generator, self).__init__()
        self.res_num = res_num
        
        model = []
        
        # Encoder
        model += [block_A(first_ch, 64), block_B(64, 128), block_B(128, 256)]
        
        # Transformer
        for _ in range(res_num):
            model += [Res_block()]
        
        # Decoder
        model += [block_B(256, 128, True), block_B(128, 64, True), block_A(64, last_ch, True)]
        
                        
        
        self.model = nn.Sequential(*model)
        
        
        
    def forward(self, x):
        
        return self.model(x)
    
    
    
class Discriminator(nn.Module):
    def __init__(self, last_ch):
        super(Discriminator, self).__init__()
        
        model = []
        model +=[
            Dis_block(last_ch, 64, first=True),
            Dis_block(64, 128),
            Dis_block(128, 256),
            Dis_block(256, 512),
            Dis_block(512, 1, stride=1, last=True),
            
        ]
        
        self.model = nn.Sequential(*model)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        y = self.model(x)
        y = torch.squeeze(y, 2)
        y = torch.squeeze(y, 2)
        y = self.sigmoid(y)
        
        return y    
    
    


    
# if __name__ == "__main__":
# #     x = torch.rand((1, 1, 256, 256)).to(DEVICE)
# #     generator = Generator(6,1,3).to(DEVICE)
# # #     discriminator = Discriminator(1).to(DEVICE)

# #     print("G(x) shape:", generator(x).shape)
# # #     print("D(x) shape:", discriminator(x).shape)



#     import numpy as np
#     import os
#     from torchsummary import summary
    
    
#     os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#     # Select GPU device number
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"

#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#     generator = Generator(6,1,3).to(DEVICE)
#     summary(generator, input_size=(1, 512, 512))