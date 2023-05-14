import numpy as np
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import torch
import os
import datetime
import wandb
# from torchsummary import summary

# import CycleGAN_fromPhaseStain
import GenXtoY_DisY

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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"




# Notation for individual wandb log name
NOTES = 'PhaseStain-CycleGAN'

# Define network hyperparameters:
HPARAMS = {
    'BATCH_SIZE': 8,
    'NUM_WORKERS': 1,
    'EPOCHS_NUM': 1500,
    'LR_D': 0.0001,
    'LR_G': 0.001,
    "BETA1": 0.9
}


# for model save, use time for data name variation
START_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# wandb init; set mode = 'disabled' annotation if you want log train/test data
wandb.init(project='CycleStain - R amp pretrain XtoY',
           config=HPARAMS,
           name='2000ep_02.sto_R_PNG_lrG(0.001)_lfD(0.0001)_256',
#            mode='disabled',
           notes=NOTES)


        

def main():

    
    model = GenXtoY_DisY.CycleGAN(HPARAMS)
    
#     model.apply(weights_init).state_dict()
    
    model = model.to(DEVICE) # get model to gpu enviornment
    
    transformer = transforms.Compose([
                                                           
                                      transforms.ToTensor(),
#                                         transforms.RandomHorizontalFlip(),  
#                                         transforms.RandomVerticalFlip(),
                                        transforms.CenterCrop((256, 256))
#                                     transforms.Resize((256,256))
                                      
    ])    
    
#     
    train_input_02 = datasets.ImageFolder(root='/data/JS/JS_dataset/CycleGAN/Pretrain_GenA/0206_R_1ch/Phase/02_R_600x600_1ch', transform=transformer)
#     train_input_06 = datasets.ImageFolder(root='/data/JS/JS_dataset/CycleGAN/Pretrain_GenA/Train_06.sto_NIR_phase', transform=transformer)
    
    
    train_target_02 = datasets.ImageFolder(root='/data/JS/JS_dataset/CycleGAN/Pretrain_GenA/0206_R_1ch/Amp/02_R_600x600_1ch', transform=transformer)
#     train_target_06 = datasets.ImageFolder(root='/data/JS/JS_dataset/CycleGAN/Pretrain_GenA/Train_06.sto_rgb', transform=transformer)
    
    
    # val
    test_input_01 = datasets.ImageFolder(root='/data/JS/JS_dataset/CycleGAN/Pretrain_GenA/Val_ground/Phase', transform=transformer)
    test_output_01 = datasets.ImageFolder(root='/data/JS/JS_dataset/CycleGAN/Pretrain_GenA/Val_ground/Amp', transform=transformer)

    # print("Data: ", dataset)
    # print("Labels: ", labels)
    
#     print(train_A[0][0].shape)






    
    train_input_02_loader = torch.utils.data.DataLoader(train_input_02,
                                                   batch_size=HPARAMS['BATCH_SIZE'],
                                                   num_workers=HPARAMS['NUM_WORKERS'],
                                                   shuffle=False
                                                   )
#     train_input_06_loader = torch.utils.data.DataLoader(train_input_06,
#                                                   batch_size=HPARAMS['BATCH_SIZE'],
#                                                   num_workers=HPARAMS['NUM_WORKERS'],
#                                                   shuffle=False

#                                                   )
    train_target_02_loader = torch.utils.data.DataLoader(train_target_02,
                                                   batch_size=HPARAMS['BATCH_SIZE'],
                                                   num_workers=HPARAMS['NUM_WORKERS'],
                                                   shuffle=False
                                                   )
#     train_target_06_loader = torch.utils.data.DataLoader(train_target_06,
#                                                   batch_size=HPARAMS['BATCH_SIZE'],
#                                                   num_workers=HPARAMS['NUM_WORKERS'],
#                                                   shuffle=False

#                                                   )
    
    
    
    
    test_input_01_loader = torch.utils.data.DataLoader(test_input_01,
                                                   batch_size=HPARAMS['BATCH_SIZE'],
                                                   num_workers=HPARAMS['NUM_WORKERS'],
                                                   shuffle=False

                                                   )
    test_target_01_loader = torch.utils.data.DataLoader(test_output_01,
                                               batch_size=HPARAMS['BATCH_SIZE'],
                                               num_workers=HPARAMS['NUM_WORKERS'],
                                                shuffle=False
                                              )
    
    
    
    # Loss
    criterion_GAN = nn.BCEWithLogitsLoss().to(DEVICE)
    label_real = torch.full((HPARAMS['BATCH_SIZE'], 1), 1, dtype=torch.float32, device=DEVICE)
    label_fake = torch.full((HPARAMS['BATCH_SIZE'], 1), 0 , dtype=torch.float32, device=DEVICE)
    
    # optimizer
    optimizer_G_x = torch.optim.Adam(model.parameters(), lr=HPARAMS["LR_G"], betas=(HPARAMS["BETA1"], 0.999))
    
    lr_scheduler_G_x = torch.optim.lr_scheduler.StepLR(optimizer_G_x, step_size=100, gamma=0.5)
    
    
    
    # wandb gpu environment log start
    wandb.watch(model, log='all')

    
    # get necessary functions to call
    
    print("----- Start Training -----")
    
    
    for epoch in range(HPARAMS['EPOCHS_NUM']):
        
       
        # Train  
        model.set_train()
        n=0
        for phase, target in zip(train_input_02_loader, train_target_02_loader):
#         for data in train_input_02_loader:
            
            phase, target = phase[0][:,0,:,:].unsqueeze(1), target[0]

#             target = torch.cat((target, target, target), 1)

            model.set_input(phase, target)
            
            
#             model.set_input(phase[0], target[0])

            
            
            model.to_device(DEVICE)
            n+=1
            model.update(n)
            train_X, train_target, train_fake_Y= model.get_output()
#             print(n)
        
#         _, _, _, loss_c, train_X, train_Y, train_fake_X, train_fake_Y, msSSIM_x2y, msSSIM_y2x = model.get_output()
        
#             _, _, loss_c, train_X, train_Y, train_fake_X, train_fake_Y = model.get_output()
        
#         for phase, target in zip(train_input_06_loader, train_target_06_loader):

            
#             # For PNG
# #             trainA = train_dataA[0][:,0,:,:].unsqueeze(1)
# #             trainB = train_dataB[0][:,0,:,:].unsqueeze(1)
# #             model.set_input(trainA, trainB)
#             model.set_input(phase[0], target[0])
            
            
            
# #             print(f"trainA: {trainA.shape}")
# #             print(f"trainB: {trainB.shape}")

# #             print(f"trainA_max: {torch.max(trainA
# #             print(f"trainA_min: {torch.min(trainA)}")
# #             print(f"trainB_max: {torch.max(trainB)}")
# #             print(f"trainB_min: {torch.min(trainB)}")
            
            
#             model.to_device(DEVICE)
#             n+=1
#             model.update(n)
# #             print(n)
        
# #         _, _, _, loss_c, train_X, train_Y, train_fake_X, train_fake_Y, msSSIM_x2y, msSSIM_y2x = model.get_output()
        
#             train_X, train_target, train_fake_Y= model.get_output()
        
#         print(f"//Train_Epoch: {epoch+1}//")
#         #print("[Epoch: {}] Loss_D: {:.6f}, Loss_G_X: {:.6f}, Loss_G_Y: {.6f}, Loss_C: {.6f}".format(epoch+1, loss_D, loss_G_X, loss_G_Y, loss_C))
# #         print(f"msSSIM(x2y): {msSSIM_x2y}\t msSSIM(y2x): {msSSIM_y2x}")
        
        
        
        # Test
        
        
        
        with torch.no_grad():
            model.set_eval()
            for phase, target in zip(test_input_01_loader, test_target_01_loader):
#             for data in valid_dataloader:
                
                phase, target = phase[0][:,0,:,:].unsqueeze(1), target[0]
#                 phase, target = torch.unsqueeze(data[:,0,:,:], 1), torch.unsqueeze(data[:,1,:,:], 1)
#                 target = torch.cat((target, target, target), 1)
                model.set_input(phase, target)
#                 data_X_phase = 1.0 - test_dataA[0]
                
                # For PNG
#                 testA = test_dataA[0][:,0,:,:].unsqueeze(1)
#                 testB = test_dataB[0][:,0,:,:].unsqueeze(1)
#                 model.set_input(testA, testB)


                
                
                model.to_device(DEVICE)
                model.update(n)
             
            # t_loss_D, t_loss_G_X, t_loss_G_Y, t_loss_C, test_X, test_Y, test_fake_X, test_fake_Y, _, _ = model.get_output()   
            test_X, test_target, test_fake_Y= model.get_output()
        
        
        print(f"//Test_Epoch: {epoch+1}//")
        print()  
        
        # lr schedular
        model.lr_schedular()
        
            
        # Train result logging    
        log_train_X = []
        log_train_target = []
#         log_train_fake_X = []
        log_train_fake_Y = []
        
        
        for i in range(1):  
            #idx_class = train_pred[i].item()
            
            # for image logging, a function MiniImagenet has 'z' with categorical index list.
            # use 'caption' for target labeling with predicted output from softmax(CrossEntropy)
            log_train_X.append(wandb.Image(train_X[i],
                                               caption='train_A'))
            log_train_target.append(wandb.Image(train_target[i],
                                               caption='train_B'))
#             log_train_fake_X.append(wandb.Image(train_fake_X[i],
#                                                caption='train_fake_A'))
            log_train_fake_Y.append(wandb.Image(train_fake_Y[i],
                                               caption='train_fake_B'))
            

        wandb.log({"Train: X": log_train_X, "Train: target": log_train_target,  "Train: Fake Y": log_train_fake_Y}, step=epoch)

        
        # Train result logging    
        log_test_X = []
        log_test_target = []
#         log_test_fake_X = []
        log_test_fake_Y = []
        
        
        for i in range(3):  
            log_test_X.append(wandb.Image(test_X[i],
                                               caption='train_A'))
            log_test_target.append(wandb.Image(test_target[i],
                                               caption='train_B'))
#             log_test_fake_X.append(wandb.Image(test_fake_X[i],
#                                                caption='train_fake_A'))
            log_test_fake_Y.append(wandb.Image(test_fake_Y[i],
                                               caption='train_fake_B'))

        wandb.log({"Test: X": log_test_X, "Test: Fake Y": log_test_fake_Y, "Test: target": log_test_target}, step=epoch)
        
        # Save parameters
        if (epoch+1) % 100 == 0:
            model.save_generator(epoch+1)
    
if __name__ == "__main__":
#     import torch
#     print(torch.cuda.is_available())
#     print(torch.cuda.device_count())
#     print(torch.cuda.get_device_name(torch.cuda.current_device()))
    main()
#     PATH = "/data/JS/Virtual_Staining/parameters/"
#     torch.save(generator.state_dict(), PATH + 'CycleGAN_state_dict.pt')