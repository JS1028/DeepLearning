import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Select GPU device number
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import torch
import datetime
import wandb
from torch import nn
from torch import optim
from torchsummary import summary
from torch.utils.data.sampler import SubsetRandomSampler

##########################################
import VGG
import ResNet

# GPU
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

###########################################

NOTES = '3_VGG_cifar10'

# Define network hyperparameters:
HPARAMS = {
    'BATCH_SIZE': 128,
    'NUM_WORKERS': 1,
    'EPOCHS_NUM': 20
}

START_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

wandb.init(project='basic_week2_VGG',
           #entity='oisl',
           config=HPARAMS,
           name=START_DATE,
           #mode='disabled',
           notes=NOTES)

def main():
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),   # (32, 32, 64) // padding=4 -> top, bottom, left, right에서 한 번씩 crop
        transforms.RandomHorizontalFlip(),      # horizontally flip with p=0.5(default)
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = datasets.CIFAR10(root='/data/deep_learning_study/', 
                                train=True, 
                                transform=transform_train,
                                download=False)

    test_dataset = datasets.CIFAR10(root='/data/deep_learning_study/', 
                               train=False, 
                               transform=transform_test)
    
            
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=HPARAMS['BATCH_SIZE'], 
                                           shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                              batch_size=HPARAMS['BATCH_SIZE'], 
                                              shuffle=False)

    class_names = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
   
    
    ###################################################
    # model = VGG.VGG_16()
    model = ResNet.ResNet_50()
    model = model.to(DEVICE)
    
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.9, weight_decay=0.0005)
    
    #############################################3#
    
    # wandb gpu environment log start
    wandb.watch(model, criterion=loss_function, log='all')
    
    
    
    # print(len(train_loader.dataset), len(test_loader.dataset)) -> 50000, 10000
    
    for epoch in range(HPARAMS['EPOCHS_NUM']):
        # Train & Test
        
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        
        for data, target in train_loader:
        
            train_input, target = data.to(DEVICE), target.to(DEVICE)
           
            optimizer.zero_grad()
            output = model(train_input)    # (128, 10)
            #print(output.shape)
            loss = loss_function(output, target)  
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            _, pred = torch.max(output, 1) # (128,)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_pred = pred.cpu().detach().numpy()
            
        print("Train_Epoch: {} // Accuracy: ({}/{}) {:.2f}%".format(epoch+1, train_correct, len(train_loader.dataset), 100. * train_correct/len(train_loader.dataset)))
          
        # Test
        model.eval()
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                test_input, target = data.to(DEVICE), target.to(DEVICE)
                
             
               
                output = model(test_input)   # (128, 10)
                test_loss += loss_function(output, target).item()
                
                _, pred = torch.max(output, 1) # (128,)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_pred = pred.cpu().detach().numpy()
                # logging, numpy로 바꿀 시 GPU에서 CPU로 꼭 바꿔주어야 한다.
                
        
        print("Test_Epoch: {} // Accuracy: ({}/{}) {:.2f}% \n".format(epoch+1, test_correct, len(test_loader.dataset), 100. * test_correct/len(test_loader.dataset)))
               
            
            
            
        # Train result logging
        save_loss = train_loss / len(train_loader)
        save_acc = 100. * train_correct / len(train_loader.dataset)
        
        log_train_label = []
     
        for i in range(80):
            idx_class = train_pred[i].item()
            
            log_train_label.append(wandb.Image(train_input[i],
                                               caption=class_names[int(idx_class)]))
        wandb.log({"Train Loss": save_loss, "Train Accuracy": save_acc, "Train input/Predict": log_train_label}, step=epoch)

        
        # Test result logging
        save_test_loss = test_loss / len(test_loader)
        save_test_acc = 100. * test_correct / len(test_loader.dataset)
        
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