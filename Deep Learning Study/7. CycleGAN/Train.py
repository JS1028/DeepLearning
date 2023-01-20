import numpy as np
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import torch
import os
import datetime
import wandb
from torchsummary import summary
import CycleGAN

from PIL import Image

from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.nn.functional as F
from torch import optim




os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Select GPU device number
os.environ["CUDA_VISIBLE_DEVICES"] = "1"




# Notation for individual wandb log name
NOTES = 'CycleGAN'

# Define network hyperparameters:
HPARAMS = {
    'BATCH_SIZE': 5,
    'NUM_WORKERS': 1,
    'EPOCHS_NUM': 100,
    'LR_D': 0.00005,
    'LR_G': 0.00005,
    "BETA1": 0.9
}





# for model save, use time for data name variation
START_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# wandb init; set mode = 'disabled' annotation if you want log train/test data
wandb.init(project='week7_CycleGAN',
           config=HPARAMS,
           name=START_DATE,
           mode='disabled',
           notes=NOTES)

# Use main function for .py operation
def main():
    """Main process function."""  # Initialize model
    

    
    
    
            
    model = CycleGAN.CycleGAN(HPARAMS)
    
    model = model.to(DEVICE) # get model to gpu enviornment
    
    model.summary((3,256,256))
    
    '''
    ######################## mine ###########################
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=HPARAMS['LR'])
   
    #########################################################
    '''
    
    transformer = transforms.Compose([
                                                           
                                      transforms.ToTensor(),
                                      transforms.Resize((256, 256))
                                      
    ])

   
    train_A = datasets.ImageFolder(root='/data/deep_learning_study/cyclegan/dataset/trainA', transform=transformer)
    train_B = datasets.ImageFolder(root='/data/deep_learning_study/cyclegan/dataset/trainB', transform=transformer)
    test_A = datasets.ImageFolder(root='/data/deep_learning_study/cyclegan/dataset/testA', transform=transformer)
    test_B = datasets.ImageFolder(root='/data/deep_learning_study/cyclegan/dataset/testB', transform=transformer)
    # print("Data: ", dataset)
    # print("Labels: ", labels)
    

    train_A_loader = torch.utils.data.DataLoader(train_A,
                                               batch_size=HPARAMS['BATCH_SIZE'],
                                               num_workers=HPARAMS['NUM_WORKERS']
                                               
                                               )
    train_B_loader = torch.utils.data.DataLoader(train_B,
                                              batch_size=HPARAMS['BATCH_SIZE'],
                                              num_workers=HPARAMS['NUM_WORKERS']
                                           
                                              )
    
    test_A_loader = torch.utils.data.DataLoader(test_A,
                                               batch_size=HPARAMS['BATCH_SIZE'],
                                               num_workers=HPARAMS['NUM_WORKERS']

                                               )
    test_B_loader = torch.utils.data.DataLoader(test_B,
                                               batch_size=HPARAMS['BATCH_SIZE'],
                                               num_workers=HPARAMS['NUM_WORKERS']
                                              
                                              )
   
    
    # wandb gpu environment log start
    wandb.watch(model, log='all')

    
    # get necessary functions to call
    
    
   
    for epoch in range(HPARAMS['EPOCHS_NUM']):
        
       
        # Train 
        model.set_train()
        for trainA, trainB in zip(train_A_loader, train_B_loader):
            
            model.set_input(trainA, trainB)
            model.to_device(DEVICE)
            
            model.update()
    
        
        _, _, _, _, train_X, train_Y, train_fake_X, train_fake_Y = model.get_output()
        
        print("//Train_epoch: {}//".format(epoch+1))
        #print("[Epoch: {}] Loss_D: {:.6f}, Loss_G_X: {:.6f}, Loss_G_Y: {.6f}, Loss_C: {.6f}".format(epoch+1, loss_D, loss_G_X, loss_G_Y, loss_C))
        
        
        # Test
        
        
        
        with torch.no_grad():
            model.set_eval()
            for testA, testB in zip(test_A_loader, test_B_loader):
                
                model.set_input(testA, testB)
                model.to_device(DEVICE)
                
                model.update()
             
            t_loss_D, t_loss_G_X, t_loss_G_Y, t_loss_C, test_X, test_Y, test_fake_X, test_fake_Y = model.get_output()   
               
        
        print("//Test_Epoch: {}//".format(epoch+1))
          
            
            
        # Train result logging    
        log_train_X = []
        log_train_Y = []
        log_train_fake_X = []
        log_train_fake_Y = []
        
        
        for i in range(5):  
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
            

        wandb.log({"Train: X": log_train_X, "Train: Y": log_train_Y, "Train: Fake X": log_train_fake_X, "Train: Fake Y": log_train_fake_Y}, step=epoch)

        
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

        wandb.log({"Test: X": log_train_X, "Test: Y": log_train_Y, "Test: Fake X": log_train_fake_X, "Test: Fake Y": log_train_fake_Y}, step=epoch)
    # Network save for inference
    save_filename = "./{}.pth".format(START_DATE)

if __name__ == "__main__":
    main()