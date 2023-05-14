import numpy as np
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import torch
import os
import datetime
import wandb
from torchsummary import summary

# import CycleGAN_fromPhaseStain
import CycleStain_semi

from Dataset import Make_Dataset

from PIL import Image
from glob import glob
import scipy.io as io

from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.nn.functional as F
from torch import optim




os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Select GPU device number
os.environ["CUDA_VISIBLE_DEVICES"] = "1"




# Notation for individual wandb log name
NOTES = 'Pretrained GAN-CycleGAN'

# Define network hyperparameters:
HPARAMS = {
    'BATCH_SIZE': 1,
    'NUM_WORKERS': 1,
    'EPOCHS_NUM': 300,
    'LR_D': 0.00001,
    ################0##############
    'LR_G_y2x': 0.0005,
    'LR_G_x2y': 0.0005,
    ##############################
    "BETA1": 0.9
}





# for model save, use time for data name variation
START_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# wandb init; set mode = 'disabled' annotation if you want log train/test data
wandb.init(project='CycleStain-semi_NIR_0206',
           config=HPARAMS,
           name='Trial 9(Sup 200_02.sto, content loss of last conv layer): 80,0.1,0,1,1 - G_lr: 0.0005 D_lr: e^-5_0.5 decay per 100_256x256',
#            mode='disabled',
           notes=NOTES)





# Use main function for .py operation
def main():
    """Main process function."""  # Initialize model
    
    '''
    Paired_Phase_data = []
    Paired_RGB_Amp_data = []
    
    UnPaired_Phase_data = []
    UnPaired_RGB_Amp_data = []
    
    contents1 = glob('/data/FPM_com/tissue NIR dataset/02.sto_S_15-71892_sec1/20221216_S_15-71892_sec1_Dataset/*.mat')
    contents2 = glob('/data/FPM_com/tissue NIR dataset/06.sto_S_10-805_sec1/S_10-805_Dataset/*.mat')
    
    contents3 = glob('/data/FPM_com/tissue NIR dataset/01.sto_S_21-11519_sec3/20221122_S_21-11519_sec3_Dataset/*.mat')
    
    # 02.sto - train
    for i, content in enumerate(contents1):
        content = io.loadmat(content)

        a = np.transpose(content['obj'], (2,0,1))
   

        # Paired
        if i < 400:
            Paired_RGB_Amp_data.append(a[:3,:,:])
            Paired_Phase_data.append(np.expand_dims(a[7,:,:], axis=0))
            UnPaired_RGB_Amp_data.append(a[:3,:,:])
        # Unpaired
        else:
            
            UnPaired_Phase_data.append(np.expand_dims(a[7,:,:], axis=0))


        if(i%50==0):
            print(i)
#     print("len: ", str(len(UnPaired_RGB_Amp_data)))
    print("02.sto done")
    print()
            
    # 06.sto - train
    for content1, content2 in zip(contents1, contents2):
        content1 = io.loadmat(content1)
        content2 = io.loadmat(content2)

        a1 = np.transpose(content1['obj'], (2,0,1))
        a2 = np.transpose(content2['obj'], (2,0,1))

        # Unpaired
#         UnPaired_RGB_Amp_data.append(a1[:3,:,:])
        UnPaired_Phase_data.append(np.expand_dims(a2[7,:,:], axis=0))

    print("06.sto done")
    
    i=0
    # 01.sto - test
    for content1, content3 in zip(contents1, contents1):
        content1 = io.loadmat(content1)
        content3 = io.loadmat(content3)

        a1 = np.transpose(content1['obj'], (2,0,1))
        a3 = np.transpose(content3['obj'], (2,0,1))
        
        # Unpaired
#         UnPaired_RGB_Amp_data.append(a1[:3,:,:])
        UnPaired_Phase_data.append(np.expand_dims(a2[7,:,:], axis=0))
        
        i +=1
        if i == 5:
            break

    print("01.sto done")
    
    num = len(UnPaired_RGB_Amp_data)
    while (1):
        for i, content in enumerate(contents1):
            content = io.loadmat(content)

            a = np.transpose(content['obj'], (2,0,1))
            

            # Paired
            if i < 400:
                
                UnPaired_RGB_Amp_data.append(a[:3,:,:])
                num +=1
                if num == len(UnPaired_Phase_data):
                    break
            # Unpaired
            else:
                break
                
        if num == len(UnPaired_Phase_data):
            break
                
    
    
    Paired_data = np.concatenate((Paired_Phase_data, Paired_RGB_Amp_data), axis=1) # (551*4, 4, 256, 256) for 256x256, (551, 4, 256, 256) for 600x600
    Paired_data = np.transpose(Paired_data, (0,2,3,1))
    
    
    
    UnPaired_data = np.concatenate((UnPaired_Phase_data, UnPaired_RGB_Amp_data), axis=1) # (551*4, 4, 256, 256) for 256x256, (551, 4, 256, 256) for 600x600
    UnPaired_data = np.transpose(UnPaired_data, (0,2,3,1))
    

    UnPaired_train_data, UnPaired_valid_data = UnPaired_data[:-5], UnPaired_data[-5:]
    
    print(np.shape(Paired_data))
    print(np.shape(UnPaired_train_data))
    print(np.shape(UnPaired_valid_data))
    
    ## Paired용과 Unpaired용을 나눠야 하나?
    Paired_dataset = Make_Dataset(Paired_data)
    UnPaired_dataset = Make_Dataset(UnPaired_train_data)
    valid_dataset = Make_Dataset(UnPaired_valid_data)

    Paired_train_loader = torch.utils.data.DataLoader(Paired_dataset, batch_size=HPARAMS['BATCH_SIZE'], shuffle=True)
    UnPaired_train_loader = torch.utils.data.DataLoader(UnPaired_dataset, batch_size=HPARAMS['BATCH_SIZE'], shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=HPARAMS['BATCH_SIZE'], shuffle=True)
    '''
    #####################################################################################################

    
            
    model = CycleStain_semi.CycleGAN(HPARAMS)
    

    
    model = model.to(DEVICE) # get model to gpu enviornment
    

    #####################################################################################################
    
    transformer1 = transforms.Compose([
                                                           
                                      transforms.ToTensor(),
#                                         transforms.RandomHorizontalFlip(),  
#                                         transforms.RandomVerticalFlip(),
                                          transforms.CenterCrop((256, 256))
                                      
    ])
    transformer2 = transforms.Compose([
                                                           
                                      transforms.ToTensor(),
                                        transforms.RandomHorizontalFlip(),  
                                        transforms.RandomVerticalFlip(),
                                          transforms.RandomCrop((256, 256))
                                      
    ])   
    
    # Supervised
    train_input_sup = datasets.ImageFolder(root='/data/JS/JS_dataset/CycleStain_semi/Supervised/Input', transform=transformer1) # 02.sto 200개
    train_target_sup = datasets.ImageFolder(root='/data/JS/JS_dataset/CycleStain_semi/Supervised/Target', transform=transformer1) # 02.sto 200개
    # Unsupervised
    train_input_unsup = datasets.ImageFolder(root='/data/JS/JS_dataset/CycleStain_semi/Unsupervised/Input', transform=transformer2) # 02.sto 200개
    train_target_unsup = datasets.ImageFolder(root='/data/JS/JS_dataset/CycleStain_semi/Unsupervised/Target', transform=transformer2) # 02.sto 200개

    # Test
    test_input_01 = datasets.ImageFolder(root='/data/JS/JS_dataset/CycleGAN/Pretrain_GenA/Test_01.sto_NIR_phase', transform=transformer2) # 5
    test_target_01 = datasets.ImageFolder(root='/data/JS/JS_dataset/CycleGAN/AIAlpha_600x600/Test_5', transform=transformer2) # 5
    

    # Supervised
    train_input_sup_loader = torch.utils.data.DataLoader(train_input_sup,
                                                   batch_size=HPARAMS['BATCH_SIZE'],
                                                   num_workers=HPARAMS['NUM_WORKERS'],
                                                   shuffle=False
                                                   )

    train_target_sup_loader = torch.utils.data.DataLoader(train_target_sup,
                                                   batch_size=HPARAMS['BATCH_SIZE'],
                                                   num_workers=HPARAMS['NUM_WORKERS'],
                                                   shuffle=False
                                                   )
    
    
    
    # Unsupervised
    train_input_unsup_loader = torch.utils.data.DataLoader(train_input_unsup,
                                                   batch_size=HPARAMS['BATCH_SIZE'],
                                                   num_workers=HPARAMS['NUM_WORKERS'],
                                                   shuffle=True
                                                   )

    train_target_unsup_loader = torch.utils.data.DataLoader(train_target_unsup,
                                                   batch_size=HPARAMS['BATCH_SIZE'],
                                                   num_workers=HPARAMS['NUM_WORKERS'],
                                                   shuffle=True
                                                   )

    
    
    
    
    test_input_01_loader = torch.utils.data.DataLoader(test_input_01,
                                                   batch_size=HPARAMS['BATCH_SIZE'],
                                                   num_workers=HPARAMS['NUM_WORKERS'],
                                                   shuffle=True

                                                   )
    test_target_01_loader = torch.utils.data.DataLoader(test_target_01,
                                               batch_size=HPARAMS['BATCH_SIZE'],
                                               num_workers=HPARAMS['NUM_WORKERS'],
                                                shuffle=True
                                              )
   
    
    # wandb gpu environment log start
    wandb.watch(model, log='all')

    
    # get necessary functions to call
    
    print("----- Start Training -----")
    
    
    for epoch in range(HPARAMS['EPOCHS_NUM']):
        
        
        ################################################################################################################################################
        # Train  
        model.set_train()
        
        
        n=0
        
        # Supervised
#         for data in Paired_train_loader:
        for phase, target in zip(train_input_sup_loader, train_target_sup_loader):
#             phase, target = torch.unsqueeze(data[:,0,:,:], 1), data[:,1:4,:,:] 
#             model.set_input(phase, target, True)
    
            model.set_input(phase[0], target[0], True)
            
            model.to_device(DEVICE)
            n+=1

            
            model.update(n)

        loss_c, train_X, train_Y, train_fake_X, train_fake_Y = model.get_output()
            
        n=0
        # Unsupervised
        for phase, target in zip(train_input_unsup_loader, train_target_unsup_loader):
#         for data in UnPaired_train_loader:          
#             phase, target = torch.unsqueeze(data[:,0,:,:], 1), data[:,1:4,:,:]
#             model.set_input(phase, target, False)

            model.set_input(phase[0], target[0], False)
        
            model.to_device(DEVICE)
            n+=1
#             if (n+1)%100 == 0:
#                 print(n)
            
            model.update(n)

#         _, _, loss_c, train_X, train_Y, train_fake_X, train_fake_Y = model.get_output()
            
            
        print(f"//Train_Epoch: {epoch+1}//")
        ################################################################################################################################################
        
        ################################################################################################################################################
        # Test
        with torch.no_grad():
            model.set_eval()
#             for data in valid_dataloader:       
            for phase, target in zip(test_input_01_loader, test_target_01_loader):

#                 phase, target = torch.unsqueeze(data[:,0,:,:], 1), data[:,1:4,:,:] 
#                 model.set_input(phase, target, False)
                model.set_input(phase[0], target[0], False)
                
                model.to_device(DEVICE)
                
                model.update(n)
            
            t_loss_G_X, t_loss_G_Y, t_loss_C, test_X, test_Y, test_fake_X, test_fake_Y= model.get_output()

        
        print(f"//Test_Epoch: {epoch+1}//")
        print()  
        ################################################################################################################################################
        
        
        # lr schedular
        model.lr_schedular()
        
            
        # Train result logging    
        log_train_X = []
        log_train_Y = []
        log_train_fake_X = []
        log_train_fake_Y = []
        
        
        for i in range(1):  
            #idx_class = train_pred[i].item()
            
            # for image logging, a function MiniImagenet has 'z' with categorical index list.
            # use 'caption' for target labeling with predicted output from softmax(CrossEntropy)
            log_train_X.append(wandb.Image(train_X[i],
                                               caption='train_A'))
            log_train_Y.append(wandb.Image(train_Y[i],
                                               caption='train_B'))
            log_train_fake_X.append(wandb.Image(train_fake_X[i],
                                               caption='train_fake_A'))
            log_train_fake_Y.append(wandb.Image(train_fake_Y[i],
                                               caption='train_fake_B'))
            

        wandb.log({"Train: X": log_train_X, "Train: Fake X": log_train_fake_X, "Train: Y": log_train_Y,  "Train: Fake Y": log_train_fake_Y, "Consistency Loss(L1)": loss_c}, step=epoch)

        
        # Train result logging    
        log_test_X = []
        log_test_Y = []
        log_test_fake_X = []
        log_test_fake_Y = []
        
        
        for i in range(1):  
            log_test_X.append(wandb.Image(test_X[i],
                                               caption='train_A'))
            log_test_Y.append(wandb.Image(test_Y[i],
                                               caption='train_B'))
            log_test_fake_X.append(wandb.Image(test_fake_X[i],
                                               caption='train_fake_A'))
            log_test_fake_Y.append(wandb.Image(test_fake_Y[i],
                                               caption='train_fake_B'))

        wandb.log({"Test: X": log_test_X, "Test: Fake X": log_test_fake_X, "Test: Y": log_test_Y,  "Test: Fake Y": log_test_fake_Y}, step=epoch)
    
    
        # Save parameters
        if (epoch+1) % 10 == 0:
            model.save_generator(epoch+1)
        
        
    
if __name__ == "__main__":
    main()
#     PATH = "/data/JS/Virtual_Staining/parameters/"
#     torch.save(generator.state_dict(), PATH + 'CycleGAN_state_dict.pt')