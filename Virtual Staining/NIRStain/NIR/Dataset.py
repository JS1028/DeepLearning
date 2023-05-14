import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import Dataset

import numpy as np

class Make_Dataset(Dataset):
    def __init__(self, data):
        self.data = data # raw 3d numpy array
        self.transform = transforms.Compose([
#             transforms.RandomCrop((256, 256)),  # crop to size (256, 256) from (600, 600)
#             transforms.RandomHorizontalFlip(),  # Flips
#             transforms.RandomVerticalFlip()
            # transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5, 0.5))
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
#         tensor_data = torch.tensor(self.data[idx, :, :, :], dtype=torch.float)
        # print(self.data.shape)
        # print(f'idx: {idx}')
        tensor_data = self.transform(self.data[idx, :, :, :]) # -> batch_size, 4, 256, 256

        tensor_data = tensor_data.type(torch.FloatTensor)

        ############################################## 
        ## For Dataset
#         phase = torch.unsqueeze(tensor_data[0,:,:], 0)
#         rgb = tensor_data[1:,:,:]
#         return phase, rgb

        ### For unstained
        return tensor_data
#     
        ############################################## 
    
