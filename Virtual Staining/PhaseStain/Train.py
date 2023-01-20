import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from glob import glob
import os
import argparse

import datetime
import wandb
import scipy.io as io
import numpy as np

from Dataset import *
from GAN import *


os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# hardware acceleration config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


HPARAMS = {
    'BATCH_SIZE': 5,
    'NUM_WORKERS': 1,
    'EPOCHS_NUM': 200,
    'gen_LR': 0.0001,
    'dis_LR': 0.00001
}


NAME = f"ep{HPARAMS['EPOCHS_NUM']},batch{HPARAMS['BATCH_SIZE']},gen_lr{HPARAMS['gen_LR']},disc_lr{HPARAMS['dis_LR']}"

NOTES = 'Virtual_Staining'
START_DATE = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

wandb.init(project='Virtual_Staining',
           config=HPARAMS,
           name=NAME,
           mode='disabled',
           notes=NOTES)






R_Phase_data = []
RGB_Amp_data = []
contents = glob('/data/FPM_com/tissue NIR dataset/01.sto_S_21-11519_sec3/20221122_S_21-11519_sec3_Dataset/*.mat')    
i=0
for content in contents:
    content = io.loadmat(content)
    a = np.transpose(content['obj'], (2,0,1))
    RGB_Amp_data.append(a[:3,44:300,44:300])
    RGB_Amp_data.append(a[:3,44:300,300:556])
    RGB_Amp_data.append(a[:3,300:556,44:300])
    RGB_Amp_data.append(a[:3,300:556,300:556])
    
    R_Phase_data.append(np.expand_dims(a[4,44:300,44:300], axis=0))
    R_Phase_data.append(np.expand_dims(a[4,44:300,300:556], axis=0))
    R_Phase_data.append(np.expand_dims(a[4,300:556,44:300], axis=0))
    R_Phase_data.append(np.expand_dims(a[4,300:556,300:556], axis=0))
    
    i+=1
    if(i%50==0):
        print(i)
#     print(np.shape(RGB_Amp_data))
#     print(np.shape(R_Phase_data))
        

# data 양이 적어서 600x600 ROI -> 4개의 256x256으로 나눔
# 총 551*4 개의 image

data = np.concatenate((R_Phase_data, RGB_Amp_data), axis=1) # (551*4, 4, 256, 256)
print(np.shape(data))


valid_split = int(len(data) * 0.9)
train_data, valid_data = data[:valid_split], data[valid_split:]

dataset = Make_Dataset(train_data)
valid_dataset = Make_Dataset(valid_data)

train_loader = DataLoader(dataset, batch_size=HPARAMS['BATCH_SIZE'], shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=HPARAMS['BATCH_SIZE'], shuffle=True)

print(f'Dataset size: {len(dataset)} | Validation dataset size: {len(valid_dataset)}\n')


# initialize models
generator = Generator()
discriminator = Discriminator()

generator = generator.to(device)
discriminator = discriminator.to(device)

# initialize optimizer witih losses
optim_g = optim.Adam(generator.parameters(), lr=HPARAMS['gen_LR'])
optim_d = optim.Adam(discriminator.parameters(), lr=HPARAMS['dis_LR'])

criterion = nn.BCELoss()
losses_g, losses_d = [], []
val_losses_g, val_losses_d = [], []

# scheduler for discriminator
# no optimized model using schduler untill now. Scheduler works as below
# lr = lr * scheduler_return_value
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optim_d, lr_lambda=lambda epoch: 1, last_epoch=-1, verbose=False)


# loss function for the discriminator
def disc_loss_func(d_gen, d_label):
    # squared error
    return d_gen ** 2 + (1 - d_label) ** 2

def compute_total_variation_loss(img, weight):
    # https://discuss.pytorch.org/t/implement-total-variation-loss-in-pytorch/55574
    bs_img, c_img, h_img, w_img = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

# loss function for the generator
def gen_loss_func(z_label, gx_input, d_gen):
    # z_label : brightfield image of the histologically stained tissue
    # gx_input: G(x_input), generated image
    # d_gen: Discriminator(gx_input) score
    # lambda, alpha: regularization parameters <- 0.02, 2000
    

    lamb, alpha = 0.02, 2000
    
    # QQ why not nn.L1loss
#     l1_loss = (z_label - gx_input).abs()  # .sum()
#     l1_loss = 256 * torch.mean(l1_loss, (1, 2, 3)).unsqueeze(1)
    l1_loss = nn.L1Loss()
    total_variation_reg = compute_total_variation_loss(gx_input, lamb)
    disc_loss = alpha * (1 - d_gen) ** 2

    # l1_loss: pixelwise L1 distance of shape [batch_size, 1]
    # total_variation_reg: single tv value over all batch
    # disc_loss: discriminator loss (to be increased) of shape [batch_size, 1]
    return l1_loss(z_label, gx_input), total_variation_reg, disc_loss






# discriminator: training function
def train_discriminator(optimizer, data_real, data_fake, infer):
    # optimizer: optimizer for the discriminator
    # data_real: historical brightfield image
    # data_fake: generated 3D channel image
    # infer: forwarding mode. no backprop if set to True
    
    if infer != 0: # inference mode
        # do not track the gradients, save hardware resources
        with torch.no_grad():
            output_real = discriminator(data_real)
            output_fake = discriminator(data_fake)
            loss = disc_loss_func(output_fake, output_real).mean()
    
    else: # training mode
        optimizer.zero_grad()

        output_real = discriminator(data_real)
        output_fake = discriminator(data_fake)
        loss = disc_loss_func(output_fake, output_real).mean()

        loss.backward()
        optimizer.step()

        # test adding scheduler (deprecated)
        scheduler.step()

    return loss, output_real, output_fake

# generator: training function
def train_generator(optimizer, data_real, data_fake, infer=False):
    # optimizer: optimizer for the generator 
    # data_real: historical brightfield image
    # data_fake: generated 3D channel image
    # infer: forwarding mode. no backprop if set to True
    
    if infer:
        # do not track the gradients, save hardware resources
        with torch.no_grad():
            d_gen = discriminator(data_fake)
            l1, tv, dl = [l.mean() for l in gen_loss_func(data_real, data_fake, d_gen)]
    else:
        optimizer.zero_grad()
        d_gen = discriminator(data_fake)
        l1, tv, dl = [l.mean() for l in gen_loss_func(data_real, data_fake, d_gen)]
        (l1 + tv + dl).backward()
        optimizer.step()
    
    return l1, tv, dl


wandb.watch(generator, criterion=gen_loss_func, log='all')
wandb.watch(discriminator, criterion=disc_loss_func, log='all')



# train generator gen_k times for one discriminator training
# gen_k = 2

epochs = HPARAMS['EPOCHS_NUM']

iter_num = 0
w = 0
v = 0
# training start
for epoch in range(epochs):
    loss_g, loss_d, loss_l1 = 0, 0, 0 # track epoch loss
    generator.train()
    discriminator.train()
    
    for idx, data in enumerate(train_loader):
        iter_num += 1
        if (iter_num+1) % 500 == 0:
            w +=1
            
        phase, rgb = [t.to(device) for t in data]
        print(f"Phase: {phase.shape}")
        print(f"RGB: {rgb.shape}")
        data_fake = generator(phase)
        data_real = rgb
        
        
        
        temp_loss_d, d_real, d_gen = train_discriminator(optim_d, data_real, data_fake.detach(), infer=v)
        
        # save losses
        loss_d += temp_loss_d
        l1, tv, dl = train_generator(optim_g, data_real, data_fake)
        v += 1
        if v == np.max(np.array(5, np.floor(7-w/2.0).astype(int))):
            v = 0
        
        loss_g += (l1 + tv + dl)
        loss_l1 += l1

    # loss logging
    # epoch_loss_d, epoch_loss_g, epoch_loss_l1 = loss_d / idx, loss_g / idx, loss_l1 / idx
    epoch_loss_d, epoch_loss_g, epoch_loss_l1 = loss_d / (idx+1), loss_g / (idx+1), loss_l1 / (idx+1)
    


    print(f'Epoch {(epoch+1):3d} of {epochs} | Generator loss: {epoch_loss_g:.0f}(L1: {epoch_loss_l1:.2f}), Discriminator loss: {epoch_loss_d:.4f}')

    # log    
    train_unstain = []
    train_fake_stain = []
    train_target_stain = []
        
        
    for i in range(3):
        train_unstain.append(wandb.Image(phase[i]))
        train_fake_stain.append(wandb.Image(data_fake[i]))
        train_target_stain.append(wandb.Image(data_real[i]))
            
    wandb.log({"Train_unstain": train_unstain, "Fake_train_Stain": train_fake_stain, "Target_train_Stain": train_target_stain,"Generator_loss": epoch_loss_g, "L1 Loss": epoch_loss_l1, "Discriminator_loss": epoch_loss_d}, step=epoch)
        
        
        
#######################################################################################################################################################        
        

    # validation
    loss_g, loss_d, loss_l1 = 0, 0, 0
    generator.eval()
    discriminator.eval()
    for idx, data in enumerate(valid_dataloader):
        phase, rgb = [t.to(device) for t in data]
        data_fake = generator(phase)
        data_real = rgb

        temp_loss_d, d_real, d_gen = train_discriminator(optim_d, data_real, data_fake.detach(), infer=True)
        loss_d += temp_loss_d

        l1, tv, dl = train_generator(optim_g, data_real, data_fake, infer=True)
        loss_g += (l1 + tv + dl)
        loss_l1 += l1
    
    # save & log loss information
    epoch_loss_d, epoch_loss_g, epoch_loss_l1 = loss_d / (idx+1), loss_g / (idx+1), loss_l1 / (idx+1)
    val_losses_g.append(epoch_loss_g)
    val_losses_d.append(epoch_loss_d)

    print(f'      Validation | Generator loss: {epoch_loss_g:.0f}(L1: {epoch_loss_l1:.2f}), Discriminator loss: {epoch_loss_d:.4f}')
    
    
    
    # log
    test_unstain = []
    test_fake_stain = []
    test_target_stain = []
        
        
    for i in range(1):
        
        test_unstain.append(wandb.Image(phase[i]))
        test_fake_stain.append(wandb.Image(data_fake[i]))
        test_target_stain.append(wandb.Image(data_real[i]))
            
    wandb.log({"Test_unstain": test_unstain, "Fake_val_Stain": test_fake_stain, "Target_val_stain": test_target_stain}, step=epoch)
        
        
    