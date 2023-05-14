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
from torch.utils.data import DataLoader

from PIL import Image


import datetime
import wandb
import scipy.io as io
import numpy as np

from Test_Dataset import *
import Create_GAN


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Select GPU device number
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


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






generator = Create_GAN.Generator(4, first_ch=1, last_ch=3)
# discriminator = Discriminator()


PATH = "/data/JS/Virtual_Staining/parameters/CycleStain_R_amp/Trial 27_best/"
generator.load_state_dict(torch.load(PATH + 'Trial 2_Pretrained_CycleStain_150ep_0206-AIAlpha_NIR_mat_256x256.pt'))

generator = generator.to(device)
generator.eval()



Phase_data = []
RGB_Amp_data = []
contents = glob('/data/FPM_com/tissue NIR dataset/02.sto_S_15-71892_sec1/20221216_S_15-71892_sec1_Dataset/*.mat')    
contents = sorted(contents)

transform = transforms.ToPILImage()



for i, content in enumerate(contents):
    if content == '/data/FPM_com/tissue NIR dataset/08.sto_P_15-71892_oiled_sec1/15-71892_Paraffin_oiled_NIR_fin/data/0220_1402_mat/full_results_processed.mat':
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
    Phase_data.append(np.expand_dims(a[7,:,:], axis=0))
    
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






## For Unstained
for idx, data in enumerate(data_loader):
    ## If input = PNG
    data = data.to(device)
    phase, rgb = torch.unsqueeze(data[:,0,:,:], 1), data[:,1:,:,:]
    # phase = phase[:,0,:,:].unsqueeze(1).to(device)
    data_fake = generator(phase)
    
    ## If input = mat
#     phase = data.to(device)
    # phase = phase[:,0,:,:].unsqueeze(1).to(device)
#     data_fake = generator(phase)
    
    ############################################## 
    ### For PNG
    
    # save_image(rgb[0], "/data/FPM_com/Virtually Stained tissue dataset/CylceStain_R_amp/NIR/Train-02_NIR_mat_600x600/02.sto_S_Trial27/Label/%.3d.png" %(idx+1))
    save_image(data_fake[0], "/data/FPM_com/Virtually Stained tissue dataset/CylceStain_R_amp/NIR/Train-02_NIR_mat_600x600/02.sto_S_Trial27_150ep/Output/%.3d.png" %(idx+1))
#     save_image(phase[0]/torch.max(phase[0]), f"/data/FPM_com/Virtually Stained tissue dataset/PhaseStain/Train-02&06_NIR_mat_600x600/Test_01_toPNG/Input/{idx+1}.png")
    

    
    
    ### For mat file
#     x = data_fake[0].detach().cpu().numpy()
#     io.savemat(f"/data/FPM_com/Virtually Stained tissue dataset/PhaseStain/Train-02&06_NIR_mat_600x600/Test_01_toMAT/{i+1}.mat", {"obj": x})
    ##############################################
    
    
    
    if (idx+1)%50==0:        
        print(f"{idx+1} done!")
#         print(torch.max(data_fake))
        print(data_fake.shape)
        
        
        
        
    
