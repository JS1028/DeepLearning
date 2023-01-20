import numpy as np
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
import torch
import os
import datetime
import wandb
from torchsummary import summary

from torchvision.transforms.functional import to_pil_image

import DCGAN

import cv2
import imageio

import gif_maker
from gif_maker import make_gif
from PIL import Image

from torch.utils.data.sampler import SubsetRandomSampler

import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.autograd import Variable





os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# Select GPU device number
os.environ["CUDA_VISIBLE_DEVICES"] = "2"




# Notation for individual wandb log name
NOTES = 'DCGAN'

# Define network hyperparameters:
HPARAMS = {
    'BATCH_SIZE': 30,
    'NUM_WORKERS': 1,
    'EPOCHS_NUM': 50,
    'LR': 0.00005
}





# for model save, use time for data name variation
START_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# wandb init; set mode = 'disabled' annotation if you want log train/test data
wandb.init(project='week6_DCGAN',
           config=HPARAMS,
           name=START_DATE,
           #mode='disabled',
           notes=NOTES)
'''
# make GIF
def make_gif(frame_folder):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.JPG")]
    frame_one = frames[0]
    frame_one.save("FAKE_MNIST.gif", format="GIF", append_images=frames,
               save_all=True, duration=100, loop=0)
'''

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 120)
        self.fc3 = nn.Linear(120, 84)
        self.fc4 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x).
        
        return x

# Use main function for .py operation
def main():
    """Main process function."""  # Initialize model

            
    gen = DCGAN.Generator()
    dis = DCGAN.Discriminator()
    
    gen = gen.to(DEVICE) # get model to gpu enviornment
    dis = dis.to(DEVICE)
 
    ######################## mine ###########################
    loss_function = nn.BCELoss()
    optimizer_gen = optim.Adam(gen.parameters(), lr=HPARAMS['LR'])
    optimizer_dis = optim.Adam(dis.parameters(), lr=HPARAMS['LR'])
    
    class_names = ('0', '1', '2', '3', '4', '5', '6', '7', '8', '9')
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
   
    MNIST = datasets.MNIST(root='data', train=True, download=True, transform=transformer)
    
    MNIST_loader = torch.utils.data.DataLoader(MNIST, batch_size=HPARAMS['BATCH_SIZE'], num_workers=HPARAMS['NUM_WORKERS'], shuffle=True)
                                                  
    # wandb gpu environment log start
    wandb.watch(gen, criterion=loss_function, log='all')
    wandb.watch(dis, criterion=loss_function, log='all')
    
    plot_z = torch.randn((30, 100, 1, 1), device=DEVICE)
    
    fake_MNIST_gif = []
    
    for epoch in range(HPARAMS['EPOCHS_NUM']):

        # Train
        for batch_idx, (images, targets) in enumerate(MNIST_loader):
            images = images.to(DEVICE)
            
            real_GT = Variable(torch.Tensor(images.size(0), 1, 1, 1).fill_(1.0), requires_grad=False)
            fake_GT = Variable(torch.Tensor(images.size(0), 1, 1, 1).fill_(0.0), requires_grad=False)
           
            
            
            real_GT = real_GT.to(DEVICE)
            fake_GT = fake_GT.to(DEVICE)
            
            #real_MNIST = Variable(images.type(torch.Tensor))a
            #real_MNIST = real_MNIST.to(DEVICE)
            real_MNIST = images
            
            
            
            # Train Discriminator
            z1 = torch.randn((images.size(0), 100, 1, 1), device=DEVICE)
            fake_dis_MNIST = gen(z1)
            
            
            loss_fake = loss_function(dis(fake_dis_MNIST), fake_GT)
            loss_real = loss_function(dis(real_MNIST), real_GT)
            loss_D = loss_fake + loss_real
            
            dis.zero_grad()
            loss_D.backward()
            optimizer_dis.step()
            
            # Train Generator
            z2 = torch.randn((images.size(0), 100, 1, 1), device=DEVICE)
            
            fake_MNIST = gen(z2)
            
            loss_G = loss_function(dis(fake_MNIST), real_GT)
            
            gen.zero_grad()
            loss_G.backward()
            optimizer_gen.step()
            
            
            
            
            
        print("[Epoch: {}]: loss_G: {:.6f}, loss_D:{:.6f}".format(epoch+1, loss_G, loss_D))
        
################################################################################################        
        
        # log
        log_fake_MNIST = []
        
        
        for i in range(30):
            mnist = gen(plot_z)[i]
            
            #print(mnist.size())
            log_fake_MNIST.append(wandb.Image(mnist))
            
        wandb.log({"Fake MNIST": log_fake_MNIST, "Generator_loss": loss_G, "Discriminator_loss": loss_D}, step=epoch)
        
        
##############################################################################################     
        
        mnist_Image = to_pil_image(mnist).convert('P')
        fake_MNIST_gif.append(mnist_Image)
        
        '''
        mnist_Image = wandb.Image(mnist)
        fake_MNIST_gif.append(mnist_Image)
        '''
        
        
    fake_MNIST_gif[0].save('./fake_MNIST_gif.gif', save_all=True, append_images=fake_MNIST_gif[1:], optimize=False, duration=100, loop=0)
    wandb.log({"GIF_MNIST": wandb.Video('./fake_MNIST_gif.gif', caption="GIF_MNIST", fps=100, format="gif")})
        
###########################################################################################3##
    
    
    
    
    # 여기에 classifier
    model = torch.load("./MLP_classifier.pt", map_location=DEVICE)
    
    with torch.no_grad():
        model.eval()
    
   
        _fake_MNIST = fake_MNIST.view(-1, 28*28)
    
        output = model(_fake_MNIST)  #(-1, 10)
        _, pred = torch.max(output, 1) # pred:(30,)
        
        
        pred = pred.cpu().detach().numpy()
        pred_MNIST = []
        for i in range(30):
            mnist = fake_MNIST[i]
            idx_class = pred[i].item()
            pred_MNIST.append(wandb.Image(mnist, caption=class_names[int(idx_class)]))

        wandb.log({"Pred MNIST": pred_MNIST})
    
    
    
    
    save_filename = "./{}.pth".format(START_DATE)
    
if __name__ == "__main__":
     
    
    main()
    
 