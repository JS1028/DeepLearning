import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from glob import glob
import os
import argparse
from torchvision import datasets
import torchvision

from torchvision.utils import save_image

from PIL import Image


import datetime
import wandb
import scipy.io as io
import numpy as np

from Dataset import *
from GAN import *


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Select GPU device number
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


# hardware acceleration config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HPARAMS = {
    'BATCH_SIZE': 1,
    'NUM_WORKERS': 1,

}


NOTES = 'Unstained Phase -> virtual stained RGB'

wandb.init(project='PhaseStain_result',
#             project='Virtual_Staining',
           config=HPARAMS,
           name='fromUStoS',
           mode='disabled',
           notes=NOTES)






generator = Generator()
# discriminator = Discriminator()


PATH = "/data/JS/Virtual_Staining/parameters/NIRStain/"
generator.load_state_dict(torch.load(PATH + 'NIRStain_02_R_600x600.pt'))

generator = generator.to(device)
generator.eval()


###############################################################################################################
'''
transformer = transforms.Compose([
                                                           
                                      transforms.ToTensor(),
#                                         transforms.RandomHorizontalFlip(),  
#                                         transforms.RandomVerticalFlip(),
                                          # transforms.RandomCrop((512, 512))
                                      
    ])    



## 02.sto
train_A = datasets.ImageFolder(root='/data/JS/Virtual_Staining/Results/fromUStoS/Test', transform=transformer)
# val_A = datasets.ImageFolder(root='/data/JS/JS_dataset/tissue NIR dataset/phase/Validation', transform=transformer)
# 

train_A_loader = torch.utils.data.DataLoader(train_A,
                                                   batch_size=HPARAMS['BATCH_SIZE'],
                                                   num_workers=1,
                                                   shuffle=False
                                            )
#                                                    )

# test_A_loader = torch.utils.data.DataLoader(val_A,
#                                                    batch_size=HPARAMS['BATCH_SIZE'],
#                                                    num_workers=1,
#                                                    shuffle=False

#                                                    )

'''
###########################################################################################################

Phase_data = []
RGB_Amp_data = []
contents = glob('/data/FPM_com/tissue NIR dataset/02.sto_S_15-71892_sec1/20221216_S_15-71892_sec1_Dataset/*.mat')    
contents = sorted(contents)

transform = transforms.ToPILImage()


for i, content in enumerate(contents):
    if content == '/data/FPM_com/tissue NIR dataset/04.bre_U_20-75112_sec4/NIR/data/0207_1844_mat/full_results_processed.mat':
            continue
    content = io.loadmat(content)
    
    ############################################## 
    ### For Dataset
    a = np.transpose(content['obj'], (2,0,1))

    ### Unstained
#     a = np.expand_dims(content['objPhase'], axis=0)
    
    ### 10.sto_U
#     a = np.expand_dims(content['obj'], axis=0)
#     a = np.log(a/np.abs(a)).imag
    ############################################## 
    
    
    ############################################## 
    ### For Dataset
    # 600x600
    RGB_Amp_data.append(a[:3,:,:])
    Phase_data.append(np.expand_dims(a[4,:,:], axis=0))
    
    ### Unstained
#     Phase_data.append(a)
    ############################################## 
    
    

    
    
    
    if(i%50==0):
        print(i)


############################################## 
### For Dataset
data = np.concatenate((Phase_data, RGB_Amp_data), axis=1) # (551*4, 4, 256, 256) for 256x256, (551, 4, 256, 256) for 600x600
data = np.transpose(data, (0,2,3,1))
print(np.shape(data))

### Unstained
# data = np.transpose(Phase_data, (0,2,3,1))
# print(np.shape(data))
############################################## 





# valid_split = int(len(data) * 0.95)
# train_data, valid_data = data[:valid_split], data[valid_split:]

############################################## 
# Make_Dataset.py 안에서 세부 수정

dataset = Make_Dataset(data)
############################################## 




data_loader = DataLoader(dataset, batch_size=HPARAMS['BATCH_SIZE'], shuffle=False, sampler=torch.utils.data.SequentialSampler(dataset))




###############################################################################################################
### For Dataset

# for idx, data in enumerate(data_loader):
#     phase, rgb = [t.to(device) for t in data]
# #     phase = phase[:,0,:,:].unsqueeze(1).to(device)
#     data_fake = generator(phase)
    
#     ############################################## 
#     ### For PNG

#     save_image(data_fake[0], "/data/FPM_com/Virtually Stained tissue dataset/PhaseStain/Train-02&06_NIR_mat_600x600/Test_06.sto_S_toPNG/Output/%.3d.png" %(idx+1))
# #     save_image(phase[0]/torch.max(phase[0]), f"/data/FPM_com/Virtually Stained tissue dataset/PhaseStain/Train-02&06_NIR_mat_600x600/Test_01_toPNG/Input/{idx+1}.png")
#     save_image(rgb[0], "/data/FPM_com/Virtually Stained tissue dataset/PhaseStain/Train-02&06_NIR_mat_600x600/Test_06.sto_S_toPNG/Label/%.3d.png" %(idx+1))
    

    
    
#     ### For mat file
# #     x = data_fake[0].detach().cpu().numpy()
# #     io.savemat(f"/data/FPM_com/Virtually Stained tissue dataset/PhaseStain/Train-02&06_NIR_mat_600x600/Test_01_toMAT/{i+1}.mat", {"obj": x})
#     ##############################################
    
    
    
#     if (idx+1)%50==0:        
#         print(f"{idx+1} done!")
# #         print(torch.max(data_fake))
#         print(rgb.shape)
    
    
## For Unstained
for idx, data in enumerate(data_loader):
    data = data.to(device)
    phase, rgb = torch.unsqueeze(data[:,0,:,:], 1), data[:,1:,:,:]
    # phase = phase[:,0,:,:].unsqueeze(1).to(device)
    data_fake = generator(phase)
    
    ############################################## 
    ### For PNG
    
    save_image(rgb[0], "/data/FPM_com/Virtually Stained tissue dataset/NIRStain/R/Train-02_R_mat_600x600/02.sto_S/Label/%.3d.png" %(idx+1))
    save_image(data_fake[0], "/data/FPM_com/Virtually Stained tissue dataset/NIRStain/R/Train-02_R_mat_600x600/02.sto_S/Output/%.3d.png" %(idx+1))
#     save_image(phase[0]/torch.max(phase[0]), f"/data/FPM_com/Virtually Stained tissue dataset/PhaseStain/Train-02&06_NIR_mat_600x600/Test_01_toPNG/Input/{idx+1}.png")
    

    
    
    ### For mat file
#     x = data_fake[0].detach().cpu().numpy()
#     io.savemat(f"/data/FPM_com/Virtually Stained tissue dataset/PhaseStain/Train-02&06_NIR_mat_600x600/Test_01_toMAT/{i+1}.mat", {"obj": x})
    ##############################################
    
    
    
    if (idx+1)%50==0:        
        print(f"{idx+1} done!")
#         print(torch.max(data_fake))
        print(data_fake.shape)
        
        
        
        
    
