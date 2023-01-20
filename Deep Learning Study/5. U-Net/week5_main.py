import numpy as np
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import torch
import os
import datetime
import wandb
from torchsummary import summary
import U_Net

from PIL import Image

from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.nn.functional as F
from torch import optim




os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Select GPU device number
os.environ["CUDA_VISIBLE_DEVICES"] = "2"




# Notation for individual wandb log name
NOTES = 'U_Net'

# Define network hyperparameters:
HPARAMS = {
    'BATCH_SIZE': 20,
    'NUM_WORKERS': 1,
    'EPOCHS_NUM': 100,
    'LR': 0.0002
}





# for model save, use time for data name variation
START_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# wandb init; set mode = 'disabled' annotation if you want log train/test data
wandb.init(project='week5_U_Net',
           config=HPARAMS,
           name=START_DATE,
           #mode='disabled',
           notes=NOTES)

# Use main function for .py operation
def main():
    """Main process function."""  # Initialize model
    

    
    
    
            
    model = U_Net.U_Net()
    
    model = model.to(DEVICE) # get model to gpu enviornment
    
 
    ######################## mine ###########################
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=HPARAMS['LR'])
   
    #########################################################

    
    transformer = transforms.Compose([
                                      #transforms.Pad(26),
                                      #transforms.RandomAffine(degrees=45,
                                      #                        scale=(0.7, 1.2),
                                    #                       translate=(0.5, 0.5)
                                    #                         ),
                                      transforms.ToTensor()
                                      
                                      # transforms.Resize(IMAGE_SHAPE[1:]),   # 227x227
                                      #transforms.Normalize((0.1307,),(0.3081,))
    ])

    ###
    # /share/ADMM_dataset/data_1_4/data/: real lensless image dataset -> 22299
    # /share/ADMM_dataset/data_1_4/label/: label
    dataset = datasets.ImageFolder(root='/share/ADMM_dataset/simul_padding_1_4/data/', transform=transformer)
    labels = datasets.ImageFolder(root='/share/ADMM_dataset/simul_padding_1_4/label/', transform=transformer)
    # print("Data: ", dataset)
    # print("Labels: ", labels)
    
    
    train_size = 20000
    
    indices = list(range(22384))
   # np.random.shuffle(indices)
    
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    #train_sampler = SubsetRandomSampler(train_indices)
    #test_sampler = SubsetRandomSampler(test_indices)
    '''
    train_data = torch.utils.data.Subset(dataset, train_indices)
    train_label= torch.utils.data.Subset(labels, train_indices)

    test_data = torch.utils.data.Subset(dataset, test_indices)
    test_label= torch.utils.data.Subset(labels, test_indices)
    
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=HPARAMS['BATCH_SIZE'],
                                               num_workers=HPARAMS['NUM_WORKERS']
                                               #sampler=train_sampler
                                               )
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=HPARAMS['BATCH_SIZE'],
                                              num_workers=HPARAMS['NUM_WORKERS']
#                                              sampler=test_sampler
                                              )
    
    train_label_loader = torch.utils.data.DataLoader(train_label,
                                               batch_size=HPARAMS['BATCH_SIZE'],
                                               num_workers=HPARAMS['NUM_WORKERS']
#                                               sampler=train_sampler
                                               )
    test_label_loader = torch.utils.data.DataLoader(test_label,
                                              batch_size=HPARAMS['BATCH_SIZE'],
                                              num_workers=HPARAMS['NUM_WORKERS']
#                                              sampler=test_sampler
                                              )
    '''
    
    # wandb gpu environment log start
    wandb.watch(model, criterion=loss_function, log='all')

    
    # get necessary functions to call
    
    
    
    for epoch in range(HPARAMS['EPOCHS_NUM']):
        
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

       

        train_data = torch.utils.data.Subset(dataset, train_indices)
        train_label= torch.utils.data.Subset(labels, train_indices)

        test_data = torch.utils.data.Subset(dataset, test_indices)
        test_label= torch.utils.data.Subset(labels, test_indices)

        train_loader = torch.utils.data.DataLoader(train_data,
                                                   batch_size=HPARAMS['BATCH_SIZE'],
                                                   num_workers=HPARAMS['NUM_WORKERS']
                                                   #sampler=train_sampler
                                                   )
        test_loader = torch.utils.data.DataLoader(test_data,
                                                  batch_size=HPARAMS['BATCH_SIZE'],
                                                  num_workers=HPARAMS['NUM_WORKERS']
    #                                              sampler=test_sampler
                                                  )

        train_label_loader = torch.utils.data.DataLoader(train_label,
                                                   batch_size=HPARAMS['BATCH_SIZE'],
                                                   num_workers=HPARAMS['NUM_WORKERS']
    #                                               sampler=train_sampler
                                                   )
        test_label_loader = torch.utils.data.DataLoader(test_label,
                                                  batch_size=HPARAMS['BATCH_SIZE'],
                                                  num_workers=HPARAMS['NUM_WORKERS']
    #                                              sampler=test_sampler
                                                  )
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_MSE = 0
        for data, label in zip(train_loader, train_label_loader):
            train_input = data[0].to(DEVICE)
            target = label[0].to(DEVICE)
            #train_input, target = data.to(DEVICE), target.to(DEVICE)
           
            #print("train_input: ", train_input.size())
            #print("target: ", target.size())
            
            optimizer.zero_grad()
            output = model(train_input) 
            
            #print("train_input: ", train_input.size())
            #print("output: ", output.size())
            loss = loss_function(output, target)  
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            '''
            _, pred = torch.max(output, 1) 
            print("pred: ", pred.size()) #(3, 410, 3)
            '''
            train_correct += output.eq(target.view_as(output)).sum().item()
            train_pred = output.cpu().detach().numpy()
            train_target = target.cpu().detach().numpy()
            
            train_MSE += np.sum((train_pred-train_target)**2) / (20000*3*308*410)
        
       
        train_MSE = np.sqrt(train_MSE)
        print("Train_Epoch: {} \tMSE: {:.6f}".format(epoch+1, train_MSE))
          
        # Test
        model.eval()
        test_loss = 0
        test_correct = 0
        
        test_MSE = 0
        with torch.no_grad():
            for data, label in zip(test_loader, test_label_loader):
                test_input = data[0].to(DEVICE)
                target = label[0].to(DEVICE)
                
             
               
                output = model(test_input) 
                test_loss += loss_function(output, target).item()
                
                #_, pred = torch.max(output, 1) 
                test_correct += output.eq(target.view_as(output)).sum().item()
                test_pred = output.cpu().detach().numpy()
                test_target = target.cpu().detach().numpy()
                # logging, numpy로 바꿀 시 GPU에서 CPU로 꼭 바꿔주어야 한다.
                test_MSE += np.sum((train_pred-train_target)**2) / (2384*3*308*410)
        
        test_MSE = np.sqrt(test_MSE)
        
        print("Test_Epoch: {} \tMSE: {:.6f} ".format(epoch+1, test_MSE))
               
        # Train result logging
        save_loss = train_loss / len(train_loader)
        save_acc = train_MSE
        
        log_train_pred = []
        log_train_label = []
        
        
        for i in range(20):  
            #idx_class = train_pred[i].item()
            pred = torch.Tensor(train_pred[i]).to(DEVICE)
            target = torch.Tensor(train_target[i]).to(DEVICE)
            # for image logging, a function MiniImagenet has 'z' with categorical index list.
            # use 'caption' for target labeling with predicted output from softmax(CrossEntropy)
            log_train_pred.append(wandb.Image(pred,
                                               caption='train'))
            log_train_label.append(wandb.Image(target,
                                               caption='train'))
            
        wandb.log({"Train Loss": save_loss, "Train MSE": save_acc, "Train Prediction": log_train_pred, "Train Target": log_train_label}, step=epoch)

        
        # Test result logging
        save_test_loss = test_loss / len(test_loader)
        save_test_acc = test_MSE
        
        log_test_pred = []
        log_test_label = []
        
        
        
        for i in range(4):
            #idx_class = test_pred[i].item()
            pred = torch.Tensor(test_pred[i]).to(DEVICE)
            target = torch.Tensor(test_target[i]).to(DEVICE)
            
            log_test_pred.append(wandb.Image(pred,
                                              caption='test'))
            log_test_label.append(wandb.Image(target,
                                              caption='test'))
        wandb.log({"Test Loss": save_test_loss, "Test MSE": save_test_acc, "Test Prediction": log_test_pred, "Test Target": log_test_label}, step=epoch)
        
        
        print("Train_loss: ", save_loss, ", Test_loss: ", save_test_loss)
        print("------------------------------------------------------------")
        
    # Network save for inference
    save_filename = "./{}.pth".format(START_DATE)

if __name__ == "__main__":
    main()