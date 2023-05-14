import torch
import torch.nn as nn
import Create_GAN
# import VGG16
from torchvision.models import vgg19

from torchsummary import summary

class CycleGAN(nn.Module):
    def __init__(self, CONFIG, test=False):
        super().__init__()
        self.CONFIG = CONFIG
        self.test = test
        self.model_G_x2y, self.model_G_y2x = Create_GAN.Generator(4, first_ch=1, last_ch=3), Create_GAN.Generator(4, first_ch=3, last_ch=1)
        self.model_D_x, self.model_D_y = Create_GAN.Discriminator(1), Create_GAN.Discriminator(3)
        
        ######################################
        ### Pretrained Generator (w/ 0206)
        PATH = "/data/JS/Virtual_Staining/parameters/CycleStain/CycleStain_R_amp/"
        self.model_G_x2y.load_state_dict(torch.load(PATH + 'Pretrain_XtoY/Glr0.001_Dlr0.0001_GenXtoY_of_CycleStain_Ramp_02_1ch_NIR_mat_256x256_400epoch.pt'))
        self.model_G_y2x.load_state_dict(torch.load(PATH + 'Pretrain_YtoX/Glr0.001_Dlr0.0001_GenYtoX_of_CycleStain_Ramp_02_1ch_NIR_mat_256x256_400epoch.pt'))
        
        self.model_D_x.load_state_dict(torch.load(PATH + 'Pretrain_YtoX/Glr0.001_Dlr0.0001_DisX_of_CycleStain_Ramp_02_1ch_NIR_mat_256x256_400epoch.pt'))
        self.model_D_y.load_state_dict(torch.load(PATH + 'Pretrain_XtoY/Glr0.001_Dlr0.0001_DisY_of_CycleStain_Ramp_02_1ch_NIR_mat_256x256_400epoch.pt'))
        
#         self.init_lastConv(self.model_G_x2y, 9, 1)
#         self.init_lastConv(self.model_G_y2x, 9, 1)

        ######################################
        
        ######################################
        self.vgg16 = vgg19(pretrained=True)
        self.criterion_L2 = nn.MSELoss() # l2 loss for content loss(perceptual loss)
#         self.feature_layers = {'35': 'last conv'}
        
        self.feature_layers = { '9': '2nd_maxpool',
                              '30': '5th_maxpool'}
#         self.last_Conv = {'54': 'last_Conv'}
        ######################################
        
        
        self.optimizer_D_x = torch.optim.Adam(self.model_D_x.parameters(), lr=CONFIG["LR_D"], betas=(CONFIG["BETA1"], 0.999))
        self.optimizer_D_y = torch.optim.Adam(self.model_D_y.parameters(), lr=CONFIG["LR_D"], betas=(CONFIG["BETA1"], 0.999))
        self.optimizer_G_x = torch.optim.Adam(self.model_G_x2y.parameters(), lr=CONFIG["LR_G_x2y"], betas=(CONFIG["BETA1"], 0.999))
        self.optimizer_G_y = torch.optim.Adam(self.model_G_y2x.parameters(), lr=CONFIG["LR_G_y2x"], betas=(CONFIG["BETA1"], 0.999))
        

        # G_x = G, G_y = F
        self.criterion_L1 = nn.L1Loss()
        self.criterion_GAN = nn.BCEWithLogitsLoss()

        
        self.label_real = torch.full((CONFIG["BATCH_SIZE"], 1), 1, dtype=torch.float32)
        #self.label_real = torch.full((CONFIG["BATCH_SIZE"], 1, 30, 30), 1, dtype=torch.float32)
        self.label_fake = torch.full((CONFIG['BATCH_SIZE'], 1), 0 , dtype=torch.float32)
        
        
        self.lr_scheduler_D_x = torch.optim.lr_scheduler.StepLR(self.optimizer_D_x, step_size=300, gamma=0.9)
        self.lr_scheduler_D_y = torch.optim.lr_scheduler.StepLR(self.optimizer_D_y, step_size=300, gamma=0.9)
        self.lr_scheduler_G_x = torch.optim.lr_scheduler.StepLR(self.optimizer_G_x, step_size=300, gamma=0.9)
        self.lr_scheduler_G_y = torch.optim.lr_scheduler.StepLR(self.optimizer_G_y, step_size=300, gamma=0.9)
        
#        self.lr_scheduler_D_x = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_D_x, gamma=0.9)
#        self.lr_scheduler_D_y = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_D_y, gamma=0.9)
#        self.lr_scheduler_G_x = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_G_x, gamma=0.9)
#        self.lr_scheduler_G_y = torch.optim.lr_scheduler.ExponentialLR(self.optimizer_G_y, gamma=0.9)
        
        
    def set_input(self, data_X, data_Y):
        self.data_X = data_X
#         self.data_X_phase = data_X_phase
        self.data_Y = data_Y
        
    def reset_grad(self, n):
        if n%50 == 0:
            self.set_requires_grad([self.model_D_x, self.model_D_y], True)
            self.optimizer_D_x.zero_grad()
            self.optimizer_D_y.zero_grad()
            self.optimizer_G_x.zero_grad()
            self.optimizer_G_y.zero_grad()
        else:
            self.set_requires_grad([self.model_D_x, self.model_D_y], False)
            self.optimizer_G_x.zero_grad()
            self.optimizer_G_y.zero_grad()
            
    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    
    #########################################################
    def get_features(self, x, model, layers):
        features = []
        for name, layer in enumerate(model.features.children()): # 0, conv
            x = layer(x)
            if str(name) in layers:
                features.append(x)
                
        return features
    
    def init_lastConv(self, model, block_num, conv_num):
        for _, layer in model.named_children():
            for names, layers in layer.named_children():
                if names == str(block_num):
                    for _, layerss in layers.named_children():
                        for n, conv in layerss.named_children():
                            if n == str(conv_num):
                                conv.weight.data.normal_(0.0, 0.02)
                                print(f"weight after initiate: {conv.weight[0,0,0]}")
    
    
    
    #########################################################
    
    def forward(self, n):
        self.dis_real_x = self.model_D_x(self.data_X)
        self.dis_real_y = self.model_D_y(self.data_Y)
        self.fake_y = self.model_G_x2y(self.data_X) # y_hat(x->y)
        self.fake_x = self.model_G_y2x(self.data_Y) # x_hat(y->x)
        self.dis_fake_x = self.model_D_x(self.fake_x) 
        self.dis_fake_y = self.model_D_y(self.fake_y)
        '''
        self.regen_x = self.model_G_x(self.fake_y)
        self.regen_y = self.model_G_y(self.fake_x)
        '''
        self.regen_x = self.model_G_y2x(self.fake_y)
        self.regen_y = self.model_G_x2y(self.fake_x)
        
        # self.identity_x = self.model_G_y2x(self.data_X)
        # self.identity_y = self.model_G_x2y(self.data_Y)
        
        #########################################################
        
        [self.fake_y_2nd, self.fake_y_5th] = self.get_features(self.fake_y, self.vgg16, self.feature_layers)
        [self.y_2nd, self.y_5th] = self.get_features(self.data_Y, self.vgg16, self.feature_layers)
        self.y_2nd = self.y_2nd.detach()
        self.y_5th = self.y_5th.detach()
        
        
        fake_x = torch.cat((self.fake_x, self.fake_x, self.fake_x), 1)
        data_x = torch.cat((self.data_X, self.data_X, self.data_X), 1)

        [self.fake_x_2nd, self.fake_x_5th] = self.get_features(fake_x, self.vgg16, self.feature_layers)
        [self.x_2nd, self.x_5th] = self.get_features(data_x, self.vgg16, self.feature_layers)
        self.x_2nd = self.x_2nd.detach()
        self.x_5th = self.x_5th.detach()
        
        # last conv
#         [self.fake_y_content] = self.get_features(self.fake_y, self.vgg16, self.feature_layers)
#         [self.y_content] = self.get_features(self.data_Y, self.vgg16, self.feature_layers)
#         self.fake_y_content = self.fake_y_content.detach()
#         self.y_content = self.y_content.detach()
        
#         fake_x = torch.cat((self.fake_x, self.fake_x, self.fake_x), 1)
#         data_x = torch.cat((self.data_X, self.data_X, self.data_X), 1)
#         [self.fake_x_content] = self.get_features(fake_x, self.vgg16, self.feature_layers)
#         [self.x_content] = self.get_features(data_x, self.vgg16, self.feature_layers)
#         self.fake_x_content = self.fake_x_content.detach()
#         self.x_content = self.x_content.detach()
        
        
        #########################################################
        if self.test == False:
            if n%3 == 1:
                self.buffer_realY = self.data_Y.detach()
                self.buffer_realX = self.data_X.detach()
                self.buffer_fakeY = self.fake_y.detach()
                self.buffer_fakeX = self.fake_x.detach()
            else:
                self.buffer_realY = torch.cat([self.buffer_realY, self.data_Y.detach()], dim=0)
                self.buffer_realX = torch.cat([self.buffer_realX, self.data_X.detach()], dim=0)
                self.buffer_fakeY = torch.cat([self.buffer_fakeY, self.fake_y.detach()], dim=0)
                self.buffer_fakeX = torch.cat([self.buffer_fakeX, self.fake_x.detach()], dim=0)
        
   
        
    def update(self, n):    
       
        self.forward(n)
        # Train Generator
        self.loss_cycle_x = self.criterion_L1(self.regen_x, self.data_X)
        self.loss_cycle_y = self.criterion_L1(self.regen_y, self.data_Y)
        self.loss_C = self.loss_cycle_x + self.loss_cycle_y
        
        # self.loss_identity_x = self.criterion_L1(self.identity_x, self.data_X)
        # self.loss_identity_y = self.criterion_L1(self.identity_y, self.data_Y)
        # self.loss_identity = self.loss_identity_x + self.loss_identity_y
        
        self.loss_G_X = self.criterion_GAN(self.dis_fake_y, torch.ones_like(self.dis_fake_y))
        self.loss_G_Y = self.criterion_GAN(self.dis_fake_x, torch.ones_like(self.dis_fake_x))
        
        ######################################
        # content loss
        self.loss_content = self.criterion_L2(self.fake_x_2nd, self.x_2nd) + self.criterion_L2(self.fake_x_5th, self.x_5th) + self.criterion_L2(self.fake_y_2nd, self.y_2nd) + self.criterion_L2(self.fake_y_5th, self.y_5th)
#         self.loss_content = self.criterion_L2(self.fake_x_content, self.x_content) + self.criterion_L2(self.fake_y_content, self.y_content)

        
        # Final Loss
        # self.loss_G = 80*self.loss_C + 0.1*self.loss_content + 1*self.loss_identity + 1*(self.loss_G_X + self.loss_G_Y)
        self.loss_G = 1*self.loss_C + (self.loss_G_X + self.loss_G_Y) # TV loss = 0

    
         ######################################

        
        if self.test == False:
            # Generator
            self.set_requires_grad([self.model_D_x, self.model_D_y], False)
            self.optimizer_G_x.zero_grad()
            self.optimizer_G_y.zero_grad()
            
            self.loss_G.backward(retain_graph=True)
            
            self.optimizer_G_x.step()
            self.optimizer_G_y.step()
        
        
        
            # Discriminator
            if n%3 == 0:
                self.set_requires_grad([self.model_D_x, self.model_D_y], True)

                for i in range(3):

                    # Forward again
                    self.dis_real_x = self.model_D_x(self.buffer_realX[i].unsqueeze(0))
                    self.dis_real_y = self.model_D_y(self.buffer_realY[i].unsqueeze(0))
                    self.dis_fake_x = self.model_D_x(self.buffer_fakeX[i].unsqueeze(0)) 
                    self.dis_fake_y = self.model_D_y(self.buffer_fakeY[i].unsqueeze(0))

                    if i == 0:
                        # Train Dis_X
                        self.loss_D_Y = 0.5 * (
                            self.criterion_GAN(self.dis_real_y, torch.ones_like(self.dis_real_y)) 
                            + self.criterion_GAN(self.dis_fake_y, torch.zeros_like(self.dis_fake_y))
                        )

                        # Train Dis_y
                        self.loss_D_X = 0.5 * (
                            self.criterion_GAN(self.dis_real_x, torch.ones_like(self.dis_real_x))
                            + self.criterion_GAN(self.dis_fake_x, torch.zeros_like(self.dis_fake_x))
                        )
                    else:
                        self.loss_D_Y += 0.5 * (
                            self.criterion_GAN(self.dis_real_y, torch.ones_like(self.dis_real_y)) 
                            + self.criterion_GAN(self.dis_fake_y, torch.zeros_like(self.dis_fake_y))
                        )

                        self.loss_D_X += 0.5 * (
                            self.criterion_GAN(self.dis_real_x, torch.ones_like(self.dis_real_x))
                            + self.criterion_GAN(self.dis_fake_x, torch.zeros_like(self.dis_fake_x))
                        )




                self.loss_D_X.backward()
                self.loss_D_Y.backward()

                self.optimizer_D_x.step()
                self.optimizer_D_y.step()
            

            


    def lr_schedular(self):
        self.lr_scheduler_D_x.step()
        self.lr_scheduler_D_y.step()
        self.lr_scheduler_G_x.step()
        self.lr_scheduler_G_y.step()
                
        
    def get_output(self):
        return self.loss_G_X, self.loss_G_Y, self.loss_C, self.data_X, self.data_Y, self.fake_x, self.fake_y, self.regen_x, self.regen_y

    def save_generator(self, ep):
        PATH = "/data/JS/Virtual_Staining/parameters/CycleStain_R_amp/"
        torch.save(self.model_G_x2y.state_dict(), PATH + f'Trial 2_Pretrained_CycleStain_{str(ep)}ep_0206-AIAlpha_NIR_mat_256x256.pt')


    def set_train(self):
        self.model_G_x2y.train()
        self.model_G_y2x.train()
        self.model_D_x.train() 
        self.model_D_y.train()
        self.test = False
        
        
    def set_eval(self):
        self.model_G_x2y.eval()
        self.model_G_y2x.eval()
        self.model_D_x.eval() 
        self.model_D_y.eval()
        self.test = True
        
    def to_device(self, DEVICE, model=False):
        self.model_G_x2y, self.model_G_y2x = self.model_G_x2y.to(DEVICE), self.model_G_y2x.to(DEVICE)
        self.model_D_x, self.model_D_y = self.model_D_x.to(DEVICE), self.model_D_y.to(DEVICE)
        #########################################################
        self.vgg16 = self.vgg16.to(DEVICE).eval()
        #########################################################
                                            
        if not model:
            self.data_X = self.data_X.to(DEVICE)
#             self.data_X_phase = self.data_X_phase.to(DEVICE)
            self.data_Y = self.data_Y.to(DEVICE)
            

            self.criterion_L1 = self.criterion_L1.to(DEVICE)
            self.criterion_GAN = self.criterion_GAN.to(DEVICE)
            
            #########################################################
            self.criterion_L2 = self.criterion_L2.to(DEVICE)
            #########################################################
        
    def summary(self, image_shape):
        summary(self.model_G_x2y.cuda(), input_size=image_shape)
        summary(self.model_D_x.cuda(), input_size=image_shape)
