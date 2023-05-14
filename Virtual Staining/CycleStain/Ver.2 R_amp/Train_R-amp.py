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
import CycleGAN_R_amp
import CycleGAN_R_amp_TV

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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"




# Notation for individual wandb log name
NOTES = 'Pretrained GAN-CycleGAN'

# Define network hyperparameters:
HPARAMS = {
    'BATCH_SIZE': 3,
    'NUM_WORKERS': 1,
    'EPOCHS_NUM': 5000,
    'LR_D': 0.00001,
    ##############################
    'LR_G_y2x': 0.0001,
    'LR_G_x2y': 0.0001,
    ##############################
    "BETA1": 0.9
}





# for model save, use time for data name variation
START_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# wandb init; set mode = 'disabled' annotation if you want log train/test data
wandb.init(project='CycleStain_R-amp_0206 & aialpha _ 2',
           config=HPARAMS,
           # consistency, TV, content, adversarial
           name='Fit_Trial 8_02->06_400ep_minmax norm&denorm: 1/0/0/1 - G_lr: 1e^-4 D_lr: 1e^-5_no decay_256x256',
#            mode='disabled',
           notes=NOTES)



save_mat_root = "/data/JS/Virtual_Staining/parameters/CycleStain/CycleStain_R_amp/Fit_mat/"

# Use main function for .py operation
def main():
    """Main process function."""  # Initialize model
    
    # Phase_data = []
    Phase_data = []
    RGB_Amp_data = []

    
    contents1 = glob('/data/FPM_com/tissue NIR dataset/02.sto_S_15-71892_sec1/Dataset/*.mat')
    contents2 = glob('/data/FPM_com/tissue NIR dataset/06.sto_S_10-805_sec1/S_10-805_Dataset/*.mat')
    
    for i, content in enumerate(contents1):
        content = io.loadmat(content)

        a = np.transpose(content['obj'], (2,0,1))
#         RGB_Amp_data.append(a[:3,:,:])
        
        RGB_Amp_data.append(a[:3,:300,:300])
        RGB_Amp_data.append(a[:3,:300,300:])
        RGB_Amp_data.append(a[:3,300:,:300])
        RGB_Amp_data.append(a[:3,300:,300:])
        
        # R_Amp_data.append(np.expand_dims(a[0,:,:], axis=0))

#         # Paired
#         if i < 400:
#             Paired_RGB_Amp_data.append(a[:3,:,:])
#             Paired_Phase_data.append(np.expand_dims(a[7,:,:], axis=0))
#             UnPaired_RGB_Amp_data.append(a[:3,:,:])d
#         # Unpaired
#         else:
            
#             UnPaired_Phase_data.append(np.expand_dims(a[7,:,:], axis=0))

        
        if(i+1) % 3 ==0:
            break
            
    for i, content in enumerate(contents2):
        content = io.loadmat(content)

        a = np.transpose(content['obj'], (2,0,1))
       
        # Phase_data.append(np.expand_dims(a[4,:,:], axis=0))
#         Phase_data.append(np.expand_dims(a[4,:,:], axis=0))
        Phase_data.append(np.expand_dims(a[4,:300,:300], axis=0))
        Phase_data.append(np.expand_dims(a[4,:300,300:], axis=0))
        Phase_data.append(np.expand_dims(a[4,300:,:300], axis=0))
        Phase_data.append(np.expand_dims(a[4,300:,300:], axis=0))

#         # Paired
#         if i < 400:
#             Paired_RGB_Amp_data.append(a[:3,:,:])
#             Paired_Phase_data.append(np.expand_dims(a[7,:,:], axis=0))
#             UnPaired_RGB_Amp_data.append(a[:3,:,:])
#         # Unpaired
#         else:
            
#             UnPaired_Phase_data.append(np.expand_dims(a[7,:,:], axis=0))
        if(i+1) % 3 ==0:
            break

        if(i%50==0):
            print(i)
    

#     data = np.transpose(data, (0,2,3,1))
    
    data = np.concatenate((Phase_data, RGB_Amp_data), axis=1) # (551*4, 4, 256, 256) for 256x256, (551, 4, 256, 256) for 600x600
    data = np.transpose(data, (0,2,3,1))

#     data = np.transpose(Phase_data, (0,2,3,1))
    
    print(np.shape(data))
    

    train_data, valid_data = data[:-3], data[-3:]
    
#     print(np.shape(data))
#     print(np.min(data[0]))
#     print(np.max(data[0]))


    
      
    dataset = Make_Dataset(train_data)
    valid_dataset = Make_Dataset(valid_data)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=HPARAMS['BATCH_SIZE'], shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=HPARAMS['BATCH_SIZE'], shuffle=True)

    print(f'Dataset size: {len(dataset)} | Validation dataset size: {len(valid_dataset)}\n')
    
    
    
    
    model = CycleGAN_R_amp.CycleGAN(HPARAMS)
    

    
    model = model.to(DEVICE) # get model to gpu enviornment
    

    '''
    
    transformer = transforms.Compose([
                                                           
                                      transforms.ToTensor(),
                                        transforms.RandomHorizontalFlip(),  
                                        transforms.RandomVerticalFlip(),
                                          transforms.RandomCrop((256, 256))
                                      
    ])    
    
    train_target = datasets.ImageFolder(root='/data/JS/JS_dataset/CycleGAN/AIAlpha_600x600/Train_1600', transform=transformer) # 800x2
    
    test_target = datasets.ImageFolder(root='/data/JS/JS_dataset/CycleGAN/AIAlpha_600x600/Test_5', transform=transformer) # 5
    
    train_target_loader = torch.utils.data.DataLoader(train_target,
                                                   batch_size=HPARAMS['BATCH_SIZE'],
                                                   num_workers=HPARAMS['NUM_WORKERS'],
                                                   shuffle=True
                                                   )
    
    test_target_loader = torch.utils.data.DataLoader(test_target,
                                               batch_size=HPARAMS['BATCH_SIZE'],
                                               num_workers=HPARAMS['NUM_WORKERS'],
                                                shuffle=True
                                              )

    '''
    # wandb gpu environment log start
    wandb.watch(model, log='all')

    
    # get necessary functions to call
    
    print("----- Start Training -----")
    
    
    for epoch in range(HPARAMS['EPOCHS_NUM']):
        
        ###########################################################################
        ## PNG
        
        '''
        # Train  
        model.set_train()
        
        
        n=0
        
        
        
        # 06 -> 02
        for phase, rgb in zip(train_loader, train_target_loader):
#             data = data.to(DEVICE)
#             phase, rgb = torch.unsqueeze(data[:,0,:,:], 1), data[:,1:4,:,:]

            model.set_input(phase, rgb[0])
            model.to_device(DEVICE)
            n+=1
#             print(n)
            
            model.update(n)
            
#             print(torch.max(phase))
#             print(torch.min(phase))
            

        _, _, loss_c, train_X, train_Y, train_fake_X, train_fake_Y = model.get_output()
        
        
        # Test
        with torch.no_grad():
            model.set_eval()
            for phase, rgb in zip(valid_dataloader, test_target_loader):
#                 data_X_phase = 1.0 - test_dataA[0]
#                 data = data.to(DEVICE)
#                 phase, rgb = torch.unsqueeze(data[:,0,:,:], 1), data[:,1:4,:,:]

                model.set_input(phase, rgb[0])
                model.to_device(DEVICE)
                
                model.update(n)
            
            t_loss_G_X, t_loss_G_Y, t_loss_C, test_X, test_Y, test_fake_X, test_fake_Y= model.get_output()
            
        print(f"//Test_Epoch: {epoch+1}//")
        print()  
        
        
        '''
        
        ###########################################################################
        ## MAT
        # Train  
        model.set_train()
        
        
        n=0
        
        
        
        # 06 -> 02
        for data in train_loader:
            data = data.to(DEVICE)
            phase, rgb = torch.unsqueeze(data[:,0,:,:], 1), data[:,1:4,:,:]
                
            for i in range (HPARAMS['BATCH_SIZE']):
                phase[i] = (phase[i] - torch.min(phase[i])) / (torch.max(phase[i]) - torch.min(phase[i]))
                rgb[i] = (rgb[i] - torch.min(rgb[i])) / (torch.max(rgb[i]) - torch.min(rgb[i]))
            
   
            
            model.set_input(phase, rgb)
            model.to_device(DEVICE)
            n+=1
#             print(n)
            
            model.update(n)
            
#             print(torch.max(phase))
#             print(torch.min(phase))
            

        _, _, loss_c, train_X, train_Y, train_fake_X, train_fake_Y, train_regen_x, train_regen_y = model.get_output()
        
        for i in range (HPARAMS['BATCH_SIZE']):
            train_fake_X[i] = (train_fake_X[i]) *  (torch.max(train_fake_X[i]) - torch.min(train_fake_X[i])) + torch.min(train_fake_X[i])
            train_fake_Y[i] = (train_fake_Y[i]) *  (torch.max(train_fake_Y[i]) - torch.min(train_fake_Y[i])) + torch.min(train_fake_Y[i])
            
            train_regen_x[i] = (train_regen_x[i]) *  (torch.max(train_regen_x[i]) - torch.min(train_regen_x[i])) + torch.min(train_regen_x[i])
            train_regen_y[i] = (train_regen_y[i]) *  (torch.max(train_regen_y[i]) - torch.min(train_regen_y[i])) + torch.min(train_regen_y[i])
            
        
        
        # Test
        with torch.no_grad():
            model.set_eval()
            for data in valid_dataloader:
#                 data_X_phase = 1.0 - test_dataA[0]
                data = data.to(DEVICE)
                phase, rgb = torch.unsqueeze(data[:,0,:,:], 1), data[:,1:4,:,:]
        
                for i in range (HPARAMS['BATCH_SIZE']):
                    phase[i] = (phase[i] - torch.min(phase[i])) / (torch.max(phase[i]) - torch.min(phase[i]))
                    rgb[i] = (rgb[i] - torch.min(rgb[i])) / (torch.max(rgb[i]) - torch.min(rgb[i]))

                model.set_input(phase, rgb)
                model.to_device(DEVICE)
                
                model.update(n)
            
            t_loss_G_X, t_loss_G_Y, t_loss_C, test_X, test_Y, test_fake_X, test_fake_Y, _, _= model.get_output()
            for i in range (HPARAMS['BATCH_SIZE']):
                test_fake_X[i] = (test_fake_X[i]) *  (torch.max(test_fake_X[i]) - torch.min(test_fake_X[i])) + torch.min(test_fake_X[i])
                test_fake_Y[i] = (test_fake_Y[i]) *  (torch.max(test_fake_Y[i]) - torch.min(test_fake_Y[i])) + torch.min(test_fake_Y[i])
            
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
        
#         model.lr_schedular()
        
        
        
        # tensor to mat
        if (epoch+1) % 200 == 0:
            X_mat = train_X[0].detach().cpu().numpy()
            regen_X_mat = train_regen_x[0].detach().cpu().numpy()
            
            Y_mat = train_Y[0].detach().cpu().numpy()
            fake_X = train_fake_X[0].detach().cpu().numpy()
            regen_Y_mat = train_regen_y[0].detach().cpu().numpy()
            
            io.savemat(save_mat_root + f"X/norm_denorm_X_mat_{epoch+1}.mat", {'img': X_mat})
            io.savemat(save_mat_root + f"regen_X/norm_denorm_regen_X_mat_{epoch+1}.mat", {'img': regen_X_mat})
            io.savemat(save_mat_root + f"Y/norm_denorm_Y_mat_{epoch+1}.mat", {'img': Y_mat})
            io.savemat(save_mat_root + f"fake_X/norm_denorm_fake_X_mat_{epoch+1}.mat", {'img': fake_X})
            io.savemat(save_mat_root + f"regen_Y/norm_denorm_regen_Y_mat_{epoch+1}.mat", {'img': regen_Y_mat}) 
          
            
        # Train result logging    
        log_train_X = []
        log_train_Y = []
        log_train_fake_X = []
        log_train_fake_Y = []
        log_train_regen_X = []
        log_train_regen_Y = []
        
        
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
            log_train_regen_X.append(wandb.Image(train_regen_x[i],
                                               caption='train_regen_A'))
            log_train_regen_Y.append(wandb.Image(train_regen_y[i],
                                               caption='train_regen_B'))
            

        wandb.log({"Train: X": log_train_X, "Train: Fake X": log_train_fake_X, "Train: Y": log_train_Y,  "Train: Fake Y": log_train_fake_Y, "Train: Regen X": log_train_regen_X, "Train: Regen Y": log_train_regen_Y, "Consistency Loss(L1)": loss_c}, step=epoch)

        
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
        '''
        if (epoch+1) % 10 == 0:
            model.save_generator(epoch+1)
        
        '''
    
if __name__ == "__main__":
    main()
#     PATH = "/data/JS/Virtual_Staining/parameters/"
#     torch.save(generator.state_dict(), PATH + 'CycleGAN_state_dict.pt')