import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import Dataset


class Make_Dataset(Dataset):
    def __init__(self, data):
        self.data = data # raw 3d numpy array
        self.transform = transforms.Compose([
            # transforms.RandomCrop((512, 512)),  # crop to size (512,512) from (600, 600)
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),  
            transforms.RandomVerticalFlip(),
            transforms.RandomCrop((256,256))
            # transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx, :, :, :]
        tensor_data = self.transform(data) # -> data_size, 1, 256, 256
        tensor_data = tensor_data.type(torch.FloatTensor)
        
#         print(tensor_data.shape)
        return tensor_data # Input RGB 
    
