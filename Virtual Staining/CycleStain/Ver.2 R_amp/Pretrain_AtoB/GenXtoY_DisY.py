import torch
import torch.nn as nn
import Create_GAN


from torchsummary import summary

class CycleGAN(nn.Module):
    def __init__(self, CONFIG, test=False):
        super().__init__()
        self.CONFIG = CONFIG
        self.test = test
        self.model_G_x2y = Create_GAN.Generator(4, first_ch=1, last_ch=3)
        self.model_D_y = Create_GAN.Discriminator(3)
        
        self.optimizer_D_y = torch.optim.Adam(self.model_D_y.parameters(), lr=CONFIG["LR_D"], betas=(CONFIG["BETA1"], 0.999))
        self.optimizer_G_x = torch.optim.Adam(self.model_G_x2y.parameters(), lr=CONFIG["LR_G"], betas=(CONFIG["BETA1"], 0.999))
        
#         self.model_G_x2y.apply(self.weights_init).state_dict()
#         self.model_D_y.apply(self.weights_init).state_dict()
#         print('Generator')
#         for layer in self.model_G_x2y.modules():
#             if isinstance(layer, nn.ConvTranspose2d):
#                 nn.init.normal_(layer.weight, 0.0, 1.0)
#                 nn.init.zeros_(layer.bias)
#                 print('ConvTranspose 성공')
#             elif isinstance(layer, nn.Conv2d):
#                 nn.init.normal_(layer.weight, 0.0, 1.0)
#                 nn.init.zeros_(layer.bias)
#                 print('Conv2d 성공')
                
                
#         print('Discriminator')     
#         for layer in self.model_D_y.modules():
#             if isinstance(layer, nn.ConvTranspose2d):
#                 nn.init.normal_(layer.weight, 0.0, 1.0)
#                 nn.init.zeros_(layer.bias)
#                 print('ConvTranspose 성공')
#             elif isinstance(layer, nn.Conv2d):
#                 nn.init.normal_(layer.weight, 0.0, 1.0)
#                 nn.init.zeros_(layer.bias)
#                 print('Conv2d 성공')
                
        # G_x = G, G_y = F
#         self.criterion_L1 = nn.L1Loss()
        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
#         self.MS_SSIM_x_red = MS_SSIM.MS_SSIM(True)
#         self.MS_SSIM_y_red = MS_SSIM.MS_SSIM(False)
        
#         self.label_real = torch.full((CONFIG["BATCH_SIZE"], 1), 1, dtype=torch.float32)
        #self.label_real = torch.full((CONFIG["BATCH_SIZE"], 1, 30, 30), 1, dtype=torch.float32)
#         self.label_fake = torch.full((CONFIG['BATCH_SIZE'], 1), 0 , dtype=torch.float32)
        
        
#         self.lr_scheduler_D_x = torch.optim.lr_scheduler.StepLR(self.optimizer_D_x, step_size=100, gamma=0.5)
        self.lr_scheduler_D_y = torch.optim.lr_scheduler.StepLR(self.optimizer_D_y, step_size=100, gamma=0.5)
        self.lr_scheduler_G_x = torch.optim.lr_scheduler.StepLR(self.optimizer_G_x, step_size=100, gamma=0.5)
#         self.lr_scheduler_G_y = torch.optim.lr_scheduler.StepLR(self.optimizer_G_y, step_size=100, gamma=0.5)
    
#     def weights_init(self, m):
#         if type(m) == nn.Linear:
#             nn.init.trunc_normal_(m.weight)
#             nn.init.zeros_(m.bias)
#         elif isinstance(nn, nn.Conv2d):
#             nn.init.trunc_normal_(m.weight)
#             nn.init.zeros_(nn.bias)
#         classname = m.__class__.__name__
#         if classname.find("Conv") != -1:
#             torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
#             print('layer success')
#         elif classname.find("InstanceNorm") != -1:
#             torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
#             torch.nn.init.constant_(m.bias.data, 0.0)
#             print('batch success')
    
        
    def set_input(self, data_X, data_Y):
        self.phase = data_X
        self.target = data_Y
        
#     def reset_grad(self, n):
#         if n%20 == 0:
#             self.set_requires_grad([self.model_D_x], True)
#             self.optimizer_D_x.zero_grad()
# #             self.optimizer_D_y.zero_grad()
#             self.optimizer_G_x.zero_grad()
# #             self.optimizer_G_y.zero_grad()
#         else:
#             self.set_requires_grad([self.model_D_x], False)
#             self.optimizer_G_x.zero_grad()
# #             self.optimizer_G_y.zero_grad()
            
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
        
    def forward(self, n):
#         self.dis_real_x = self.model_D_x(self.phase)
        self.dis_real_y = self.model_D_y(self.target)
        self.fake_y = self.model_G_x2y(self.phase) # y_hat(x->y)
#         self.fake_x = self.model_G_y2x(self.target) # x_hat(y->x)
#         self.dis_fake_x = self.model_D_x(self.fake_x) 
        self.dis_fake_y = self.model_D_y(self.fake_y)
        '''
        self.regen_x = self.model_G_x(self.fake_y)
        self.regen_y = self.model_G_y(self.fake_x)
        '''
#         self.regen_x = self.model_G_y2x(self.fake_y)
#         self.regen_y = self.model_G_x2y(self.fake_x)
        
#         self.identity_x = self.model_G_y2x(self.phase)
#         self.identity_y = self.model_G_x2y(self.data_Y)
        
        
        if self.test == False:
            if n%10 == 1:
                self.buffer_realY = self.target.detach()
                self.buffer_realX = self.phase.detach()
                self.buffer_fakeY = self.fake_y.detach()
#                 self.buffer_fakeX = self.fake_x.detach()
            else:
                self.buffer_realY = torch.cat([self.buffer_realY, self.target.detach()], dim=0)
#                 self.buffer_realX = torch.cat([self.buffer_realX, self.phase.detach()], dim=0)
                self.buffer_fakeY = torch.cat([self.buffer_fakeY, self.fake_y.detach()], dim=0)
#                 self.buffer_fakeX = torch.cat([self.buffer_fakeX, self.fake_x.detach()], dim=0)
        
    
#     def forward_D_real(self):
#         self.dis_real_x = self.model_D_x(self.phase)
#         self.dis_real_y = self.model_D_y(self.data_Y)
        
#     def forward_D_fake(self):
#         self.fake_y = self.model_G_x2y(self.phase) 
#         self.fake_x = self.model_G_y2x(self.data_Y)
#         self.dis_fake_x = self.model_D_x(self.fake_x)
#         self.dis_fake_y = self.model_D_y(self.fake_y)
        
#     def forward_G_XYX(self):
#         self.fake_y = self.model_G_x2y(self.phase)
#         self.dis_fake_y = self.model_D_y(self.fake_y)
#         # self.regen_x = self.model_G_x(self.fake_y)
#         self.regen_x = self.model_G_y2x(self.fake_y)
#         self.G_x2y_y = self.model_G_x2y(self.data_y)
        
#     def forward_G_YXY(self):
#         self.fake_x = self.model_G_y2x(self.data_Y)
#         self.dis_fake_x = self.model_D_x(self.fake_x)
#         # self.regen_y = self.model_G_y(self.fake_x)
#         self.regen_y = self.model_G_x2y(self.fake_x)
#         self.G_y2x_x = self.model_G_x2y(self.phase)
        
        
    def update(self, n):    
        ### NEED TO BE ADDED ###
        
        
        self.forward(n)
        # Train Generator

#         self.loss_G_X = self.criterion_GAN(self.dis_fake_y, torch.ones_like(self.dis_fake_y))
       

       
#         self.loss_G = self.loss_C + self.loss_G_X + self.loss_G_Y + 0.1*(1-self.msSSIM_x2y) + 0.1*(1-self.msSSIM_y2x)
#         self.loss_G = 10*self.loss_C + 5*self.loss_identity + self.loss_G_X + self.loss_G_Y

#         self.loss_G = self.loss_G_X + self.l1(self.target, self.fake_y).mean()
        
        if self.test == False:
            self.set_requires_grad([self.model_D_y], False)
            
                
            #########################
#             self.optimizer_G_x.zero_grad()
#             self.loss_G.backward(retain_graph=True)
#             self.optimizer_G_x.step()

            self.train_generator(self.optimizer_G_x, self.target, self.fake_y, self.dis_fake_y)
    
            #########################

        
        
        
            # Discriminator
            if n%10 == 0:
                self.set_requires_grad([self.model_D_y], True)

                for i in range(10):

                    # Forward again
#                     self.dis_real_x = self.model_D_x(self.buffer_realX[i].unsqueeze(0))
                    self.dis_real_y = self.model_D_y(self.buffer_realY[i].unsqueeze(0))
#                     self.dis_fake_x = self.model_D_x(self.buffer_fakeX[i].unsqueeze(0)) 
                    self.dis_fake_y = self.model_D_y(self.buffer_fakeY[i].unsqueeze(0))

                    if i == 0:
                        # Train Dis_X
                        self.loss_D_X = 0.5 * (
                            self.criterion_GAN(self.dis_real_y, torch.ones_like(self.dis_real_y)) 
                            + self.criterion_GAN(self.dis_fake_y, torch.zeros_like(self.dis_fake_y))
                        )

#                         # Train Dis_y
#                         self.loss_D_Y = 0.5 * (
#                             self.criterion_GAN(self.dis_real_x, torch.ones_like(self.dis_real_x))
#                             + self.criterion_GAN(self.dis_fake_x, torch.zeros_like(self.dis_fake_x))
#                         )
                    else:
                        self.loss_D_X += 0.5 * (
                            self.criterion_GAN(self.dis_real_y, torch.ones_like(self.dis_real_y)) 
                            + self.criterion_GAN(self.dis_fake_y, torch.zeros_like(self.dis_fake_y))
                        )

#                         self.loss_D_Y += 0.5 * (
#                             self.criterion_GAN(self.dis_real_x, torch.ones_like(self.dis_real_x))
#                             + self.criterion_GAN(self.dis_fake_x, torch.zeros_like(self.dis_fake_x))
#                         )

#                 if self.test == False:


                self.loss_D_X.backward()
                # self.loss_D_Y.backward()

#                 self.optimizer_D_x.step()
                self.optimizer_D_y.step()
            
        
#         if self.test == False:
#             self.reset_grad(n)
            
#             if n%20 == 0:
#                 self.loss_G.backward(retain_graph=True)
#                 self.loss_G.backward(retain_graph=True)
#                 self.loss_D_X.backward()
#                 self.loss_D_Y.backward()
                
#                 self.optimizer_G_x.step()
#                 self.optimizer_G_y.step()
#                 self.optimizer_D_x.step()
#                 self.optimizer_D_y.step()
                
#             else:
#                 self.loss_G.backward(retain_graph=True)
#                 self.loss_G.backward()
                
#                 self.optimizer_G_x.step()
#                 self.optimizer_G_y.step()

    def lr_schedular(self):
#         self.lr_scheduler_D_x.step()
        self.lr_scheduler_D_y.step()
        self.lr_scheduler_G_x.step()
#         self.lr_scheduler_G_y.step()
                
        
    def get_output(self):
#         return self.loss_D_X+self.loss_D_Y, self.loss_G_X, self.loss_G_Y, self.loss_C, self.phase, self.data_Y, self.fake_x, self.fake_y, self.msSSIM_x2y, self.msSSIM_y2x
        return self.phase, self.target, self.fake_y

    def save_generator(self, ep):
        PATH = "/data/JS/Virtual_Staining/parameters/CycleStain/CycleStain_R_amp/Pretrain_XtoY/PNG/"
        torch.save(self.model_G_x2y.state_dict(), PATH + f'Glr0.001_Dlr0.0001_GenXtoY_of_CycleStain_Ramp_02_1ch_NIR_png_256x256_{str(ep)}epoch.pt')
        torch.save(self.model_D_y.state_dict(), PATH + f'Glr0.001_Dlr0.0001_DisY_of_CycleStain_Ramp_02_1ch_NIR_png_256x256_{str(ep)}epoch.pt')


    def set_train(self):
        self.model_G_x2y.train()
#         self.model_G_y2x.train()
#         self.model_D_x.train() 
        self.model_D_y.train()
        self.test = False
        
    #####################################################################
    
    def compute_total_variation_loss(self, img, weight):
        # https://discuss.pytorch.org/t/implement-total-variation-loss-in-pytorch/55574
        bs_img, c_img, h_img, w_img = img.size()
        tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
        tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
        return weight * (tv_h + tv_w) / (bs_img * c_img * h_img * w_img)

    # loss function for the generator
    def gen_loss_func(self, z_label, gx_input, d_gen):
        # z_label : brightfield image of the histologically stained tissue
        # gx_input: G(x_input), generated image
        # d_gen: Discriminator(gx_input) score
        # lambda, alpha: regularization parameters <- 0.02, 2000


        lamb, alpha = 0.02, 2000

        # QQ why not nn.L1loss
    #     l1_loss = (z_label - gx_input).abs()  # .sum()
    #     l1_loss = 256 * torch.mean(l1_loss, (1, 2, 3)).unsqueeze(1)
        l1_loss = nn.L1Loss()
        total_variation_reg = self.compute_total_variation_loss(gx_input, lamb)
        disc_loss = alpha * (1 - d_gen) ** 2

        # l1_loss: pixelwise L1 distance of shape [batch_size, 1]
        # total_variation_reg: single tv value over all batch
        # disc_loss: discriminator loss (to be increased) of shape [batch_size, 1]
        return l1_loss(z_label, gx_input), total_variation_reg, disc_loss
    
    def train_generator(self, optimizer, data_real, data_fake, d_gen):
    # optimizer: optimizer for the generator 
    # data_real: historical brightfield image
    # data_fake: generated 3D channel image
    # infer: forwarding mode. no backprop if set to True
    
   
        optimizer.zero_grad()
        
        l1, tv, dl = [l.mean() for l in self.gen_loss_func(data_real, data_fake, d_gen)]
        (l1 + tv + dl).backward()
        optimizer.step()
    
        return l1, tv, dl
        
     #####################################################################
        
    def set_eval(self):
        self.model_G_x2y.eval()
#         self.model_G_y2x.eval()
#         self.model_D_x.eval() 
        self.model_D_y.eval()
        self.test = True
        
    def to_device(self, DEVICE, model=False):
        self.model_G_x2y = self.model_G_x2y.to(DEVICE)
        self.model_D_y = self.model_D_y.to(DEVICE)
        if not model:
            self.phase = self.phase.to(DEVICE)

            self.target = self.target.to(DEVICE)
            
#             self.label_real = torch.full((self.CONFIG['BATCH_SIZE'], 1), 1, dtype=torch.float32, device=DEVICE)
#             self.label_fake = torch.full((self.CONFIG['BATCH_SIZE'], 1), 0 , dtype=torch.float32, device=DEVICE)
#             self.criterion_L1 = self.criterion_L1.to(DEVICE)
            self.criterion_GAN = self.criterion_GAN.to(DEVICE)
            self.l1 = self.l1.to(DEVICE)
#             self.MS_SSIM_x_red = self.MS_SSIM_x_red.to(DEVICE)
#             self.MS_SSIM_y_red = self.MS_SSIM_y_red.to(DEVICE)
        
    def summary(self, image_shape):
        summary(self.model_G_x2y.cuda(), input_size=image_shape)
        summary(self.model_D_x.cuda(), input_size=image_shape)
