import numpy as np
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import torch
import os
import datetime
import wandb
from torchsummary import summary
import CNN_STN

from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.nn.functional as F




os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Select GPU device number
os.environ["CUDA_VISIBLE_DEVICES"] = "2"




# Notation for individual wandb log name
NOTES = 'CNN_STN'

# Define network hyperparameters:
HPARAMS = {
    'BATCH_SIZE': 256,
    'NUM_WORKERS': 1,
    'EPOCHS_NUM': 60,
    'LR': 0.01
}





# for model save, use time for data name variation
START_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# wandb init; set mode = 'disabled' annotation if you want log train/test data
wandb.init(project='week4_STN_CNN',
           config=HPARAMS,
           name=START_DATE,
           mode='disabled',
           notes=NOTES)

# Use main function for .py operation
def main():
    """Main process function."""  # Initialize model
    

    
    
    
            
    model = CNN_STN.Net()
    
    model = model.to(DEVICE) # get model to gpu enviornment
    
 
    ######################## mine ###########################
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=HPARAMS['LR'], momentum=0, weight_decay=0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    class_names = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
    
    #########################################################

    
    transformer = transforms.Compose([
                                      transforms.Pad(26),
                                      transforms.RandomAffine(degrees=45,
                                                              scale=(0.7, 1.2),
                                                              translate=(0.5, 0.5)
                                                             ),
                                      transforms.ToTensor(),
                                      # transforms.Resize(IMAGE_SHAPE[1:]),   # 227x227
                                      transforms.Normalize((0.1307,),(0.3081,))
    ])


    train_data = datasets.MNIST(root='data', train=True, download=True, transform=transformer)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=transformer)
    
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=HPARAMS['BATCH_SIZE'],
                                               num_workers=HPARAMS['NUM_WORKERS'],
                                               shuffle=True,
                                               sampler=None
                                               )
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=HPARAMS['BATCH_SIZE'],
                                              num_workers=HPARAMS['NUM_WORKERS'],
                                              shuffle=True,
                                              sampler=None
                                              )

    # wandb gpu environment log start
    wandb.watch(model, criterion=loss_function, log='all')

    
    # get necessary functions to call
    
    
    
    for epoch in range(HPARAMS['EPOCHS_NUM']):
        # Train & Test
        
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        
        for data, target in train_loader:
        
            train_input, target = data.to(DEVICE), target.to(DEVICE)
           
            optimizer.zero_grad()
            output = model(train_input)    # 
            #print(output.shape)
            loss = loss_function(output, target)  
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            _, pred = torch.max(output, 1) 
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_pred = pred.cpu().detach().numpy()
        
        scheduler.step()
        
        print("Train_Epoch: {} \tAccuracy: ({}/{}) {:.0f}%".format(epoch+1, train_correct, len(train_data), 100. * train_correct/len(train_data)))
          
        # Test
        model.eval()
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                test_input, target = data.to(DEVICE), target.to(DEVICE)
                
             
               
                output = model(test_input) 
                test_loss += loss_function(output, target).item()
                
                _, pred = torch.max(output, 1) 
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_pred = pred.cpu().detach().numpy()
                # logging, numpy로 바꿀 시 GPU에서 CPU로 꼭 바꿔주어야 한다.
        
        print("Test_Epoch: {} \tAccuracy: ({}/{}) {:.0f}% \n".format(epoch+1, test_correct, len(test_data), 100. * test_correct/len(test_data)))
               
        # Train result logging
        save_loss = train_loss / len(train_data)
        save_acc = 100. * train_correct / len(train_data)
        
        log_train_label = []
        
        
        for i in range(96):  
            idx_class = train_pred[i].item()
            
            # for image logging, a function MiniImagenet has 'z' with categorical index list.
            # use 'caption' for target labeling with predicted output from softmax(CrossEntropy)
            log_train_label.append(wandb.Image(train_input[i],
                                               caption=class_names[int(idx_class)]))
        wandb.log({"Train Loss": save_loss, "Train Accuracy": save_acc, "Train input/Predict": log_train_label}, step=epoch)

        
        # Test result logging
        save_test_loss = test_loss / len(test_data)
        save_test_acc = 100. * test_correct / len(test_data)
        
        log_test_label = []
        
        
        
        for i in range(16):
            idx_class = test_pred[i].item()
            
            log_test_label.append(wandb.Image(test_input[i],
                                              caption=class_names[int(idx_class)]))
        wandb.log({"Test Loss": save_test_loss, "Test Accuracy": save_test_acc, "Test input/Predict": log_test_label}, step=epoch)

    # Network save for inference
    save_filename = "./{}.pth".format(START_DATE)

if __name__ == "__main__":
    main()