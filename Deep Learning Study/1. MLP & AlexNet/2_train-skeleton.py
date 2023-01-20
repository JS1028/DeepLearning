import numpy as np
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import torch
import os
import datetime
import wandb
from torchsummary import summary
import Alex
from mimagenet import MiniImagenet
from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.nn.functional as F



# nn 과 F 사용가능?
# line244: 10->64??



os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Select GPU device number
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Notation for individual wandb log name
NOTES = '2_Alex_mimg'

# Define network hyperparameters:
HPARAMS = {
    'BATCH_SIZE': 5,
    'NUM_WORKERS': 1,
    'EPOCHS_NUM': 30,
    'LR': 0.0001
}

# Referred Alex net size (227, 227) as color image
IMAGE_SHAPE = (3, 227, 227)

# for model save, use time for data name variation
START_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# wandb init; set mode = 'disabled' annotation if you want log train/test data
wandb.init(project='basic',
           config=HPARAMS,
           name=START_DATE,
           #mode='disabled',
           notes=NOTES)

# Use main function for .py operation
def main():
    """Main process function."""  # Initialize model
    
    
    # summary(model, input_size=IMAGE_SHAPE) # check model summary
    
    
    
            
    model = Alex.Alex()
    
    model = model.to(DEVICE) # get model to gpu enviornment
    
    '''
    for p in model.parameters():
        print(p)
        break
    '''
    
    ######################## mine ###########################
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=HPARAMS['LR'], momentum=0.9, weight_decay=0.0005)
   # optimizer = torch.optim.Adam(model.parameters(), lr = HPARAMS['LR'])
    
    #########################################################
    
    
    
    
    
    
    # for m-imagenet datgaset ToTensor function is already located in MiniImagenet
    transformer = transforms.Compose([
                                      transforms.Resize(IMAGE_SHAPE[1:]),   # 227x227
                                      transforms.Normalize((0.5, 0.5, 0.5),
                                                           (0.5, 0.5, 0.5))
    ])

    # we only use mode='train' data since train target category differ from test's.
    # (dataset-driven problem exist - nevermind)
    # if you need sparse subset dataset, use mode='test'
    dataset = MiniImagenet(root='/data/deep_learning_study/miniimagenet',
                              mode='train',
                              download=False,
                              transform=transformer)
    
    train_size = 30000 # set train dataset size 30,000
    
    # dataset subsampling due to trainset/testset split.
    # if a dataset folder has sperated with train/test, just use datasets.ImageFolder
    indices = list(range(36000)) # whole dataset range from 0 to 36,000
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:train_size], indices[train_size:] # shuffle된 0~35999 수 중 앞 30000개는 for training, 나머지는 for test
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=HPARAMS['BATCH_SIZE'],
                                               num_workers=HPARAMS['NUM_WORKERS'],
                                               sampler=train_sampler
                                               )
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=HPARAMS['BATCH_SIZE'],
                                              num_workers=HPARAMS['NUM_WORKERS'],
                                              sampler=test_sampler
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
            output = model(train_input)    # (5, 64)
            #print(output.shape)
            loss = loss_function(output, target)  
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            
            _, pred = torch.max(output, 1) # (5,)
            train_correct += pred.eq(target.view_as(pred)).sum().item()
            train_pred = pred.cpu().detach().numpy()
            
        print("Train_Epoch: {} \tAccuracy: ({}/{}) {:.0f}%".format(epoch+1, train_correct, len(train_loader.sampler), 100. * train_correct/len(train_loader.sampler)))
          
        # Test
        model.eval()
        test_loss = 0
        test_correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                test_input, target = data.to(DEVICE), target.to(DEVICE)
                
             
               
                output = model(test_input) # 5x64
                test_loss += loss_function(output, target).item()
                
                _, pred = torch.max(output, 1) # (5,)
                test_correct += pred.eq(target.view_as(pred)).sum().item()
                test_pred = pred.cpu().detach().numpy()
                # logging, numpy로 바꿀 시 GPU에서 CPU로 꼭 바꿔주어야 한다.
        
        print("Test_Epoch: {} \tAccuracy: ({}/{}) {:.0f}%".format(epoch+1, test_correct, len(test_loader.sampler), 100. * test_correct/len(test_loader.sampler)))
               
        # Train result logging
        save_loss = train_loss / len(train_loader)
        save_acc = 100. * train_correct / len(train_loader.sampler)
        
        log_train_label = []
        
        
        for i in range(1):  
            idx_class = train_pred[i].item()
            
            # for image logging, a function MiniImagenet has 'z' with categorical index list.
            # use 'caption' for target labeling with predicted output from softmax(CrossEntropy)
            log_train_label.append(wandb.Image(train_input[i],
                                               caption=dataset.z[int(idx_class)]))
        wandb.log({"Train Loss": save_loss, "Train Accuracy": save_acc, "Train input/Predict": log_train_label}, step=epoch)

        
        # Test result logging
        save_test_loss = test_loss / len(test_loader)
        save_test_acc = 100. * test_correct / len(test_loader.sampler)
        
        log_test_label = []
        
        
        
        for i in range(1):
            idx_class = test_pred[i].item()
            
            log_test_label.append(wandb.Image(test_input[i],
                                              caption=dataset.z[int(idx_class)]))
        wandb.log({"Test Loss": save_test_loss, "Test Accuracy": save_test_acc, "Test input/Predict": log_test_label}, step=epoch)

    # Network save for inference
    save_filename = "./{}.pth".format(START_DATE)

if __name__ == "__main__":
    main()