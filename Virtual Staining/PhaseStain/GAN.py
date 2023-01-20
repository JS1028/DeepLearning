import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def gen_down_block(in_ch, out_ch):
            layers = [
                nn.Conv2d(in_ch, out_ch//2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=out_ch//2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(out_ch//2, out_ch//2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=out_ch//2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(out_ch//2, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=out_ch),
                nn.LeakyReLU(0.1)
            ]

            return nn.Sequential(*layers)

        def gen_up_block(in_ch):
            # decrease channel by fourfold
            assert in_ch % 4 == 0, "wrong channel config"
            out_ch = in_ch // 4
            layers = [
                nn.Conv2d(in_ch, out_ch*2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=out_ch*2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(out_ch*2, out_ch*2, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=out_ch*2),
                nn.LeakyReLU(0.1),
                nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(num_features=out_ch),
                nn.LeakyReLU(0.1)
            ]

            return nn.Sequential(*layers)

        self.pooling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.down1 = gen_down_block(1, 64)
        self.down2 = gen_down_block(64, 128)
        self.down3 = gen_down_block(128, 256)
        self.down4 = gen_down_block(256, 512)
        self.connection = gen_down_block(512, 512)

        self.unpool1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, padding=0, bias=True)
        self.unpool2 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0, bias=True)
        self.unpool3 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2, padding=0, bias=True)
        self.unpool4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2, padding=0, bias=True)

        self.up1 = gen_up_block(4 * 256)
        self.up2 = gen_up_block(4 * 128)
        self.up3 = gen_up_block(4 * 64)
        self.up4 = gen_up_block(4 * 32)
        self.recover = nn.Conv2d(32, 3, kernel_size=3, stride=1, padding=1)
           
            
        

    def forward(self, x):
        add1 = self.point_wise_add(x, self.down1(x))
        pool1 = self.pooling(add1)
        add2 = self.point_wise_add(pool1, self.down2(pool1))
        pool2 = self.pooling(add2)
        add3 = self.point_wise_add(pool2, self.down3(pool2))
        pool3 = self.pooling(add3)
        add4 = self.point_wise_add(pool3, self.down4(pool3))
        
        latent = self.connection(self.pooling(add4))

        up1 = torch.cat((self.unpool1(latent), add4), dim=1)
        up1 = self.up1(up1)

        up2 = torch.cat((self.unpool2(up1), add3), dim=1)
        up2 = self.up2(up2)

        up3 = torch.cat((self.unpool3(up2), add2), dim=1)
        up3 = self.up3(up3)

        up4 = torch.cat((self.unpool4(up3), add1), dim=1)
        up4 = self.up4(up4)

        output = self.recover(up4)

        return output

    def point_wise_add(self, x, conv_x):
        assert x.shape[1] <= conv_x.shape[1], "wrong sequence"
        
        # fine.shape[1] <= coarse.shape[1]
        padded = torch.zeros(conv_x.shape, device=conv_x.get_device())  # init tensor with 0  
        # mod_ch = x.shape[1] 
        padded[:, :x.shape[1], :, :] = x      
        # padded의 크기 = conv_x. 그 중 x 크기 만큼은 x와 동일. 나머지 부분은 0
        
        
        return padded + conv_x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def gen_disc_block(in_ch):
            layers = [
                nn.Conv2d(in_ch, in_ch, 3, padding=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(in_ch, 2*in_ch, kernel_size=3, stride=2, padding=1),
                nn.LeakyReLU(0.1)
                
            ]

            return nn.Sequential(*layers)

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU(0.1)
#             nn.Conv2d(64, 64, 3, padding=1),
#             nn.LeakyReLU(0.1),
        )

        self.disc_block1 = gen_disc_block(64)
        self.disc_block2 = gen_disc_block(128)
        self.disc_block3 = gen_disc_block(256)
        self.disc_block4 = gen_disc_block(512)
        self.disc_block5 = gen_disc_block(1024)
        self.pooling = nn.AvgPool2d(kernel_size=8)
        # input: 512x512 -> kernel_size = 16
        # input: 256x256 -> kernel_size = 8
        self.fc = nn.Sequential(
            nn.Flatten(),
            # nn.Linear(2048, 1024), # @@ 2048????
            nn.Linear(2048, 2048), 
            nn.LeakyReLU(0.1),
            # nn.Linear(1024, 1),
            nn.Linear(2048, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.disc_block1(x)
        x = self.disc_block2(x)
        x = self.disc_block3(x)
        x = self.disc_block4(x)
        x = self.disc_block5(x)
        x = self.pooling(x)
        x = self.fc(x)
        return x



# if __name__ == '__main__':
    
#     import numpy as np
#     import os
#     from torchsummary import summary
   
    
#     os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#     os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#     # hardware acceleration config
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#     gen = Discriminator().to(device)

#     summary(gen, input_size=(3, 512, 512))
    