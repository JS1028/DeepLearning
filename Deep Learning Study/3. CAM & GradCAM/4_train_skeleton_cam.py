import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Select GPU device number
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import numpy as np
import cv2
from torchvision import datasets, models
import torchvision
import torchvision.transforms as transforms
import torch
import datetime
import wandb
import traceback
from torch import nn
from torch import optim
from torchsummary import summary
from mimagenet import MiniImagenet
from torch.utils.data.sampler import SubsetRandomSampler

import CAM

# CAM
# extract cam=> 두 줄



NOTES = '5_CAM_mimg'

# Define network hyperparameters:
HPARAMS = {
    'BATCH_SIZE': 5,
    'NUM_WORKERS': 1,
    'EPOCHS_NUM': 10,
    'LR': 0.0001
}

image_shape = (3, 224, 224)

START_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

wandb.init(project='week3',
           config=HPARAMS,
           name=START_DATE,
           #mode='disabled',
           notes=NOTES)

class LossFunction(nn.Module):
    """Loss function class for multiple loss function."""

    def __init__(self):
        super(LossFunction, self).__init__()
        self.criterion_CE = nn.CrossEntropyLoss().to(DEVICE)

    def forward(self, output, label):
        CE_loss = self.criterion_CE(output, label)
        loss = CE_loss
        return loss

    
###################################################################################

def extract_cam(model, outmap, label): 
    """Extract CAM data from a map output & label match
    
        Args:
            model (nn.Module): CAM Model which can get weight from
            outmap (tensor): Output from the features of model
            label (tensor): Ground truth label
            
        Returns:
            cam (numpy arr): cam data with multiplication of output map & class weight mathed from label
            
    """
    ### should be implemented ###
    weight = model.linear.weight[int(label)] # (512)
    
    
    
    weight = weight.unsqueeze(-1)            # (512, 1)
    weight = weight.unsqueeze(-1)            # (512, 1, 1)
    # outmap: (512, 7, 7)
    
    cam = outmap*weight
    cam = torch.sum(cam, axis=0) # cam: (7,7)
    cam = cam.cpu().detach().numpy()
    
    return cam


def train(train_loader, model, loss_function, optimizer):
    # return train_input , train_label, train_map v , train_loss, train_acc, train_pred
    model.train()
    train_loss = 0
    train_correct = 0
    
    for data, target in train_loader:
        train_input, target = data.to(DEVICE), target.to(DEVICE)
                
        optimizer.zero_grad()
        output, map = model(train_input)
        
        loss = loss_function(output, target)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        
        _, pred = torch.max(output, 1) 
        train_correct += pred.eq(target.view_as(pred)).sum().item()
        train_pred = pred
    
    
    return train_input, target, map, train_loss, train_correct, train_pred

def test(test_loader, model, loss_function):
    # test_input, test_label, test_map, test_loss, test_acc, test_pred
    model.eval()
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            test_input, target = data.to(DEVICE), target.to(DEVICE)
            
            output, map = model(test_input)
            test_loss += loss_function(output, target)
            
            _, pred = torch.max(output, 1)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
            test_pred = pred
            
    return test_input, target, map, test_loss, test_correct, test_pred
    

######################################################################################

def main():
    """Main process function."""  # Initialize model
    
    model = models.vgg16(pretrained=True) # Get pretrained vgg16 for CAM feature-map extraction
    features = model.features
    model = model.to(DEVICE)
    #summary(model, input_size=image_shape)
    
    ### model CAM should be implemented ###
    model = CAM.CAM(features=features, num_classes=64, init_weights=True) # CAM class should include "features" args.
    
    model = model.to(DEVICE)
    #summary(model, input_size=image_shape)

    transformer = transforms.Compose([
                                      transforms.Resize(image_shape[1:]),
                                      transforms.Normalize((0.5, 0.5, 0.5),
                                                           (0.5, 0.5, 0.5))])

    dataset = MiniImagenet(root='/data/deep_learning_study/miniimagenet',
                              mode='train',
                              download=False,
                              transform=transformer)
    
    train_size = 30000
    indices = list(range(len(dataset)))
    np.random.shuffle(indices)
    train_indices, test_indices = indices[:train_size], indices[train_size:]
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    
    train_loader = torch.utils.data.DataLoader(dataset,
                                               batch_size=HPARAMS['BATCH_SIZE'],
                                               num_workers=HPARAMS['NUM_WORKERS'],
#                                                shuffle=True,
                                               sampler=train_sampler,
                                               )
    test_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=HPARAMS['BATCH_SIZE'],
                                              num_workers=HPARAMS['NUM_WORKERS'],
#                                               shuffle=True,
                                               sampler=test_sampler
                                              )

    loss_function = LossFunction()

    optimizer = optim.Adam(model.parameters(), lr=HPARAMS['LR'])

    wandb.watch(model, criterion=loss_function, log='all')
  

    # Training and test
    for epoch in range(HPARAMS['EPOCHS_NUM']):
        train_input, train_label, train_map, train_loss, train_acc, train_pred = train(train_loader, # no need to print output, need map
                                                                                       model,
                                                                                       loss_function,
                                                                                       optimizer)
        test_input, test_label, test_map, test_loss, test_acc, test_pred = test(test_loader,
                                                                                model,
                                                                                loss_function)

        # Logging part using tensorboard
        print('Train :: epoch: {}. loss: {:.5f}. Acc: {:.3f}'.format(epoch,
                                                                   train_loss / len(train_loader),
                                                                   train_acc / len(train_loader.sampler) * 100.))

        # Training result logging
        save_loss = train_loss / len(train_loader)
        save_acc = train_acc / len(train_loader.sampler) * 100.

        
        
        train_input = train_input[:HPARAMS["BATCH_SIZE"]].cpu()  # Recommend restricted batch size smaller than 10
        train_label = train_label[:HPARAMS["BATCH_SIZE"]].detach().cpu().numpy()
        train_pred = train_pred[:HPARAMS["BATCH_SIZE"]].detach().cpu().numpy()
        train_cam = []
        log_train_label = []
        log_train_cam = []        
        for i in range(HPARAMS["BATCH_SIZE"]):
            display = extract_cam(model, train_map[i], train_label[i])  # train_map: (5, 512,7,7) train_label: (5) display: (7, 7)
            display = cv2.resize(display, dsize=image_shape[1:], interpolation=cv2.INTER_CUBIC) # Resize from 7x7 cam image to 224x224            
            display = display * train_input.cpu().detach().numpy()[i] # cam mask mapping with input image
            display = np.transpose(display, (1,2,0)) # wandb Image get numpy as (224,224,3), torch as (3,224,224)
            
            train_cam = display
            idx_class = train_pred[i].item()
            
            log_train_label.append(wandb.Image(train_input[i],
                                               caption=dataset.z[idx_class]))
            log_train_cam.append(wandb.Image(train_cam,
                                               caption=dataset.z[idx_class]))
        wandb.log({"Train Loss": save_loss,
                   "Train Accuracy": save_acc,
                   "Train input/Predict": log_train_label,
                   "Train CAM": log_train_cam,
                  }, step=epoch)

        # Test result logging
        save_test_loss = test_loss / len(test_loader)
        save_test_acc = test_acc / len(test_loader.sampler) * 100.
        
        test_input = test_input[:HPARAMS["BATCH_SIZE"]].cpu()
        test_label = test_label[:HPARAMS["BATCH_SIZE"]].cpu()
        test_pred = test_pred[:HPARAMS["BATCH_SIZE"]].detach().cpu().numpy()
        test_cam = []
        log_test_label = []
        log_test_cam = []
        for i in range(HPARAMS["BATCH_SIZE"]):
            display = extract_cam(model, test_map[i], test_label[i])
            display = cv2.resize(display, dsize=image_shape[1:], interpolation=cv2.INTER_CUBIC)
            display = display * test_input.cpu().detach().numpy()[i]
            display = np.transpose(display, (1,2,0))
            
            test_cam = display
            idx_class = test_pred[i].item()
            
            log_test_label.append(wandb.Image(test_input[i],
                                              caption=dataset.z[idx_class]))
            log_test_cam.append(wandb.Image(test_cam,
                                               caption=dataset.z[idx_class]))
        wandb.log({"Test Loss": save_test_loss,
                   "Test Accuracy": save_test_acc,
                   "Test input/Predict": log_test_label,
                   "Test CAM": log_test_cam
                  }, step=epoch)


if __name__ == "__main__":
    main()