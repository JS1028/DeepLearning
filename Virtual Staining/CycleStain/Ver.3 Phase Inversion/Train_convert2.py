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
import CycleStain_convert2
# import CycleGAN_R_amp_TV

from Dataset import Make_Dataset
from torch.utils.data import DataLoader

from PIL import Image
from glob import glob
import scipy.io as io

from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.nn.functional as F
from torch import optim




os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Select GPU device number
os.environ["CUDA_VISIBLE_DEVICES"] = "0"




# Notation for individual wandb log name
NOTES = 'Pretrained GAN-CycleGAN'

# Define network hyperparameters:
HPARAMS = {
    'BATCH_SIZE': 4,
    'NUM_WORKERS': 1,
    'EPOCHS_NUM': 300,
    'LR_D': 0.0001,
    ##############################
    'LR_G_y2x': 0.0002,
    'LR_G_x2y': 0.0002,
    ##############################
    "BETA1": 0.9
}





# for model save, use time for data name variation
START_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# wandb init; set mode = 'disabled' annotation if you want log train/test data
wandb.init(project='CycleStain_PhaseConversion_0206 & aialpha',
           config=HPARAMS,
           # consistency, TV, content, adversarial
           name='Trial 1_5res_lastConv normal init.: 80/0/0.1/1 - G: 2e^-4 D: 1e^-4_no decay_convert phase',
           mode='disabled',
           notes=NOTES)





# Use main function for .py operation
def main():
    """Main process function."""  # Initialize model
    
    # Phase_data = []
#     Phase_data = []
#     RGB_Amp_data = []

    
#     contents1 = glob('/data/FPM_com/tissue NIR dataset/02.sto_S_15-71892_sec1/Dataset/*.mat')
#     contents2 = glob('/data/FPM_com/tissue NIR dataset/06.sto_S_10-805_sec1/S_10-805_Dataset/*.mat')
    
#     for i, content in enumerate(contents1):
#         content = io.loadmat(content)

#         a = np.transpose(content['obj'], (2,0,1))
       
#         Phase_data.append(np.invert(np.expand_dims(a[4,:,:], axis=0)))
#         # R_Amp_data.append(np.expand_dims(a[0,:,:], axis=0))

# #         # Paired
# #         if i < 400:
# #             Paired_RGB_Amp_data.append(a[:3,:,:])
# #             Paired_Phase_data.append(np.expand_dims(a[7,:,:], axis=0))
# #             UnPaired_RGB_Amp_data.append(a[:3,:,:])
# #         # Unpaired
# #         else:
            
# #             UnPaired_Phase_data.append(np.expand_dims(a[7,:,:], axis=0))


#         if(i%100==0):
#             break
            
#     for i, content in enumerate(contents2):
#         content = io.loadmat(content)

#         a = np.transpose(content['obj'], (2,0,1))
       
#         # Phase_data.append(np.expand_dims(a[4,:,:], axis=0))
#         RGB_Amp_data.append(a[:3,:,:])

# #         # Paired
# #         if i < 400:
# #             Paired_RGB_Amp_data.append(a[:3,:,:])
# #             Paired_Phase_data.append(np.expand_dims(a[7,:,:], axis=0))
# #             UnPaired_RGB_Amp_data.append(a[:3,:,:])
# #         # Unpaired
# #         else:
            
# #             UnPaired_Phase_data.append(np.expand_dims(a[7,:,:], axis=0))


#         if(i%100==0):
#             break
    

# #     data = np.transpose(data, (0,2,3,1))
    
#     data = np.concatenate((Phase_data, RGB_Amp_data), axis=1) # (551*4, 4, 256, 256) for 256x256, (551, 4, 256, 256) for 600x600
#     data = np.transpose(data, (0,2,3,1))

# #     data = np.transpose(Phase_data, (0,2,3,1))
    
#     print(np.shape(data))
    

#     train_data, valid_data = data[:-5], data[-5:]
    
# #     print(np.shape(data))
# #     print(np.min(data[0]))
# #     print(np.max(data[0]))


    
      
#     dataset = Make_Dataset(train_data)
#     valid_dataset = Make_Dataset(valid_data)

#     train_loader = torch.utils.data.DataLoader(dataset, batch_size=HPARAMS['BATCH_SIZE'], shuffle=True)
#     valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=HPARAMS['BATCH_SIZE'], shuffle=True)

#     print(f'Dataset size: {len(dataset)} | Validation dataset size: {len(valid_dataset)}\n')
    
    
    
            
    model = CycleStain_convert2.CycleGAN(HPARAMS)
    

    
    model = model.to(DEVICE) # get model to gpu enviornment
    

    
    
    transformer = transforms.Compose([
                                                           
                                      transforms.ToTensor(),
                                        transforms.RandomHorizontalFlip(),  
                                        transforms.RandomVerticalFlip(),
                                          transforms.RandomCrop((256, 256))
                                      
    ])    
    
    train_input = datasets.ImageFolder(root='/data/JS/JS_dataset/CycleGAN/Pretrain_GenA/0206_NIR_600x600_1ch/0206_NIR_phase_uint8', transform=transformer)
    train_target = datasets.ImageFolder(root='/data/JS/JS_dataset/CycleGAN/AIAlpha_600x600/Train_1600', transform=transformer) # 800x2
    
    test_input = datasets.ImageFolder(root='/data/JS/JS_dataset/CycleGAN/Pretrain_GenA/0206_NIR_600x600_1ch/02_NIR_phase_Test_uint8', transform=transformer)
    test_target = datasets.ImageFolder(root='/data/JS/JS_dataset/CycleGAN/AIAlpha_600x600/Test_5', transform=transformer) # 5
    
    
    train_input_loader = torch.utils.data.DataLoader(train_input,
                                                   batch_size=HPARAMS['BATCH_SIZE'],
                                                   num_workers=HPARAMS['NUM_WORKERS'],
                                                   shuffle=True
                                                   )
    train_target_loader = torch.utils.data.DataLoader(train_target,
                                                   batch_size=HPARAMS['BATCH_SIZE'],
                                                   num_workers=HPARAMS['NUM_WORKERS'],
                                                   shuffle=True
                                                   )
    test_input_loader = torch.utils.data.DataLoader(test_input,
                                                   batch_size=HPARAMS['BATCH_SIZE'],
                                                   num_workers=HPARAMS['NUM_WORKERS'],
                                                   shuffle=True
                                                   )
    test_target_loader = torch.utils.data.DataLoader(test_target,
                                               batch_size=HPARAMS['BATCH_SIZE'],
                                               num_workers=HPARAMS['NUM_WORKERS'],
                                                shuffle=True
                                              )

    
    # wandb gpu environment log start
    wandb.watch(model, log='all')

    
    # get necessary functions to call
    
    print("----- Start Training -----")
    
    
    for epoch in range(HPARAMS['EPOCHS_NUM']):
        
       
        # Train  
        model.set_train()
        
        
        n=0
        
        ###########################################################################
        
        # 06 -> 02
        for phase, rgb in zip(train_input_loader, train_target_loader):
            phase = phase[0][:,0,:,:].unsqueeze(1)
            phase, rgb = phase.to(DEVICE), rgb[0].to(DEVICE)
#             phase = transforms.functional.invert(phase)
#             print(torch.max(phase))
#             phase, rgb = torch.unsqueeze(data[:,0,:,:], 1), data[:,1:4,:,:]
            
#             print(phase.shape)
            model.set_input(phase, rgb)
            model.to_device(DEVICE)
            n+=1
#             print(n)
            
            model.update(n)

        _, _, loss_c, train_X, train_Y, train_fake_X, train_fake_Y = model.get_output()
        
        
        with torch.no_grad():
            model.set_eval()
            for phase, rgb in zip(test_input_loader, test_target_loader):
#                 data_X_phase = 1.0 - test_dataA[0]
                phase = phase[0][:,0,:,:].unsqueeze(1)
                
                phase, rgb = phase.to(DEVICE), rgb[0].to(DEVICE)
#                 phase = transforms.functional.invert(phase)
#                 phase, rgb = torch.unsqueeze(data[:,0,:,:], 1), data[:,1:4,:,:]
                
                model.set_input(phase, rgb)
                model.to_device(DEVICE)
                
                model.update(n)
            
            t_loss_G_X, t_loss_G_Y, t_loss_C, test_X, test_Y, test_fake_X, test_fake_Y= model.get_output()
            
        print(f"//Test_Epoch: {epoch+1}//")
        print()  
        
        
        ###########################################################################
        
        '''
        
        # 02.sto + 06.sto
        for phase, target in zip(train_loader, train_target_loader):          
            
#             print(f"trainA: {trainA.shape}")
#             print(f"trainB: {trainB.shape}")

#             print(f"trainA_max: {torch.max(trainA)}")
#             print(f"trainA_min: {torch.min(trainA)}")
#             print(f"trainB_max: {torch.max(trainB)}")
#             print(f"trainB_min: {torch.min(trainB)}")

            model.set_input(phase, target[0])
            model.to_device(DEVICE)
            n+=1
#             print(n)
            
            model.update(n)

        _, _, loss_c, train_X, train_Y, train_fake_X, train_fake_Y = model.get_output()
            
            
        print(f"//Train_Epoch: {epoch+1}//")
        #print("[Epoch: {}] Loss_D: {:.6f}, Loss_G_X: {:.6f}, Loss_G_Y: {.6f}, Loss_C: {.6f}".format(epoch+1, loss_D, loss_G_X, loss_G_Y, loss_C))
#         print(f"msSSIM(x2y): {msSSIM_x2y}\t msSSIM(y2x): {msSSIM_y2x}")
        
        
        # Test
        with torch.no_grad():
            model.set_eval()
            for phase, target in zip(valid_dataloader, test_target_loader):
#                 data_X_phase = 1.0 - test_dataA[0]

                model.set_input(phase, target[0])
                model.to_device(DEVICE)
                
                model.update(n)
            
            t_loss_G_X, t_loss_G_Y, t_loss_C, test_X, test_Y, test_fake_X, test_fake_Y= model.get_output()

        
        print(f"//Test_Epoch: {epoch+1}//")
        print()  
        
        
        '''
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
        if (epoch+1) % 20 == 0:
            model.save_generator(epoch+1)
        
        
    
if __name__ == "__main__":
    main()
#     PATH = "/data/JS/Virtual_Staining/parameters/"
#     torch.save(generator.state_dict(), PATH + 'CycleGAN_state_dict.pt')