import torch
import torch.nn as nn
import Create_GAN
from torchsummary import summary

class CycleGAN(nn.Module):
    def __init__(self, CONFIG, test=False):
        super().__init__()
        self.CONFIG = CONFIG
        self.test = test
        self.model_G_x, self.model_G_y = Create_GAN.Generator(6), Create_GAN.Generator(6)
        self.model_D_x, self.model_D_y = Create_GAN.Discriminator(), Create_GAN.Discriminator()
        self.optimizer_D_x = torch.optim.Adam(self.model_D_x.parameters(), lr=CONFIG["LR_D"], betas=(CONFIG["BETA1"], 0.999))
        self.optimizer_D_y = torch.optim.Adam(self.model_D_y.parameters(), lr=CONFIG["LR_D"], betas=(CONFIG["BETA1"], 0.999))
        self.optimizer_G_x = torch.optim.Adam(self.model_G_x.parameters(), lr=CONFIG["LR_G"], betas=(CONFIG["BETA1"], 0.999))
        self.optimizer_G_y = torch.optim.Adam(self.model_G_y.parameters(), lr=CONFIG["LR_G"], betas=(CONFIG["BETA1"], 0.999))
        # G_x = G, G_y = F
        self.criterion_L1 = nn.L1Loss()
        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.label_real = torch.full((CONFIG["BATCH_SIZE"], 1, 30, 30), 1, dtype=torch.float32)
        #self.label_real = torch.full((CONFIG["BATCH_SIZE"], 1, 30, 30), 1, dtype=torch.float32)
        self.label_fake = torch.full((CONFIG['BATCH_SIZE'], 1, 30, 30), 0 , dtype=torch.float32)
        
        
    def set_input(self, data_X, data_Y):
        self.data_X, _ = data_X
        self.data_Y, _ = data_Y
        
    def reset_grad(self):
        self.optimizer_D_x.zero_grad()
        self.optimizer_D_y.zero_grad()
        self.optimizer_G_x.zero_grad()
        self.optimizer_G_y.zero_grad()
        
    def forward(self):
        self.dis_real_x = self.model_D_x(self.data_X)
        self.dis_real_y = self.model_D_y(self.data_Y)
        self.fake_y = self.model_G_x(self.data_X) # y_hat(x->y)
        self.fake_x = self.model_G_y(self.data_Y) # x_hat(y->x)
        self.dis_fake_x = self.model_D_x(self.fake_x) 
        self.dis_fake_y = self.model_D_y(self.fake_y)
        '''
        self.regen_x = self.model_G_x(self.fake_y)
        self.regen_y = self.model_G_y(self.fake_x)
        '''
        self.regen_x = self.model_G_y(self.fake_y)
        self.regen_y = self.model_G_x(self.fake_x)
    
    def forward_D_real(self):
        self.dis_real_x = self.model_D_x(self.data_X)
        self.dis_real_y = self.model_D_y(self.data_Y)
        
    def forward_D_fake(self):
        self.fake_y = self.model_G_x(self.data_X)
        self.fake_x = self.model_G_y(self.data_Y)
        self.dis_fake_x = self.model_D_x(self.fake_x)
        self.dis_fake_y = self.model_D_y(self.fake_y)
        
    def forward_G_XYX(self):
        self.fake_y = self.model_G_x(self.data_X)
        self.dis_fake_y = self.model_D_y(self.fake_y)
        # self.regen_x = self.model_G_x(self.fake_y)
        self.regen_x = self.model_G_y(self.fake_y)
        
    def forward_G_YXY(self):
        self.fake_x = self.model_G_y(self.data_Y)
        self.dis_fake_x = self.model_D_x(self.fake_x)
        # self.regen_y = self.model_G_y(self.fake_x)
        self.regen_y = self.model_G_x(self.fake_x)
        
    def update(self):    
        ### NEED TO BE ADDED ###
        
        
        self.forward()
        # Train Generator
        self.loss_cycle_x = self.criterion_L1(self.regen_x, self.data_X)
        self.loss_cycle_y = self.criterion_L1(self.regen_y, self.data_Y)
        self.loss_C = self.loss_cycle_x + self.loss_cycle_y
        
        self.loss_G_X = self.criterion_GAN(self.dis_fake_y, torch.ones_like(self.dis_fake_y))
        self.loss_G_Y = self.criterion_GAN(self.dis_fake_x, torch.ones_like(self.dis_fake_x))
        
        self.loss_G = self.loss_C + self.loss_G_X + self.loss_G_Y
        
        
        # Train Dis_X
        self.loss_D_X = 0.5 * (
            self.criterion_GAN(self.dis_real_y, torch.ones_like(self.dis_real_y)) 
            + self.criterion_GAN(self.dis_fake_y, torch.zeros_like(self.dis_fake_y))
        )
        
        # Train Dis_y
        self.loss_D_Y = 0.5 * (
            self.criterion_GAN(self.dis_real_x, torch.ones_like(self.dis_real_x))
            + self.criterion_GAN(self.dis_fake_x, torch.zeros_like(self.dis_fake_x))
        )
        
        if self.test == False:
            self.reset_grad()

            self.loss_G.backward(retain_graph=True)
            self.loss_G.backward(retain_graph=True)
            self.loss_D_X.backward(retain_graph=True)
            self.loss_D_Y.backward()

            self.optimizer_G_x.step()
            self.optimizer_G_y.step()
            self.optimizer_D_x.step()
            self.optimizer_D_y.step()
        
        
        
    def get_output(self):
        return self.loss_D_X+self.loss_D_Y, self.loss_G_X, self.loss_G_Y, self.loss_C, self.data_X, self.data_Y, self.fake_x, self.fake_y
        
    def set_train(self):
        self.model_G_x.train()
        self.model_G_y.train()
        self.model_D_x.train() 
        self.model_D_y.train()
        self.test = False
        
        
    def set_eval(self):
        self.model_G_x.eval()
        self.model_G_y.eval()
        self.model_D_x.eval() 
        self.model_D_y.eval()
        self.test = True
        
    def to_device(self, DEVICE, model=False):
        self.model_G_x, self.model_G_y = self.model_G_x.to(DEVICE), self.model_G_y.to(DEVICE)
        self.model_D_x, self.model_D_y = self.model_D_x.to(DEVICE), self.model_D_y.to(DEVICE)
        if not model:
            self.data_X = self.data_X.to(DEVICE)
            self.data_Y = self.data_Y.to(DEVICE)
            self.label_real = torch.full((self.CONFIG['BATCH_SIZE'], 1, 30, 30), 1, dtype=torch.float32, device=DEVICE)
            self.label_fake = torch.full((self.CONFIG['BATCH_SIZE'], 1, 30, 30), 0 , dtype=torch.float32, device=DEVICE)
            self.criterion_L1 = self.criterion_L1.to(DEVICE)
            self.criterion_GAN = self.criterion_GAN.to(DEVICE)
        
    def summary(self, image_shape):
        summary(self.model_G_x.cuda(), input_size=image_shape)
        summary(self.model_D_x.cuda(), input_size=image_shape)
