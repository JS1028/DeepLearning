import numpy as np
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

# z = torch.rand(1, 100, 1, 1, device=device)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.generate = nn.Sequential(
            nn.ConvTranspose2d(in_channels=100, out_channels=256, kernel_size=7, stride=1, padding=0), # (256, 7, 7)
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1), # (128, 14, 14)
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=1, stride=1, padding=0), # (64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, stride=2, padding=1), # (1, 28, 28)
            nn.Tanh()

        )
    
    def forward(self, z):
        fake_mnist = self.generate(z)
        
        return fake_mnist
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.discriminate = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=4, stride=2, padding=1), # (32, 14, 14)
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1), # (64, 7, 7)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1), # (128, 3, 3)
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=0), # (1, 1, 1)
            nn.Sigmoid()
            
        )
        
    def forward(self, x):
        y = self.discriminate(x)
        
        return y
        