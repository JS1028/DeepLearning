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
import cv2
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch import optim
from torchsummary import summary
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt

# imsave 등 이용하여 img파일 저장하여 확인. 주어진 이미지 3개, class는 4개(개,고양이,피자,뱀), loggging x, train x. 
# imshow로는 terminal에서 못 봄.
# 사진 결과가 ppt랑 똑같아야.


NOTES = 'Grad-CAM'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

NUM_CLASSES = 1000


class VGGnet(nn.Module):

    def __init__(self, features):
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, NUM_CLASSES),
            nn.ReLU(True)
        )


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        feature = self.classifier(x)
        return feature, F.softmax(x)

# VGGNet 만들 때 사용
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1) #
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)] #
            in_channels = v
    return nn.Sequential(*layers)
    

def preprocess_image(img, resize_img=True):
    "image array to image tensor"
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if resize_img:
        img = img.resize((224,224))
    img_arr = np.float32(img) # (224, 224, 3)
    img_arr = img_arr.transpose(2, 0, 1) # (3, 224, 224)

    for c, _ in enumerate(img_arr):
        img_arr[c] /= 255
        img_arr[c] -= mean[c]
        img_arr[c] /= std[c]

    img_tensor = torch.from_numpy(img_arr).float()
    img_tensor = torch.unsqueeze(img_tensor,0) # (1, 3, 224, 224)
    
    return img_tensor


def reprocess_image(img):
    "image tensor to image array"
    img = torch.squeeze(img,0) # (3, 224, 224)
    mean = [-0.485, -0.456, -0.406]
    std = [1/0.229, 1/0.224, 1/0.225]
    
    # img_re = copy.copy(img.cpu().data.numpy())
    img_re =img.cpu().data.numpy().copy()
    for c in range(3):
        img_re[c] /= std[c]
        img_re[c] -= mean[c]
        
    img_re[img_re > 1] = 1
    img_re[img_re < 0] = 0
    img_re = np.round(img_re * 255)

    img_re = np.uint8(img_re).transpose(1, 2, 0)
    
    return img_re
    

def main():
    # Test images with ImageNet class number
    test_list = (('/data/deep_learning_study/cam_test/kingsnake.jpg', 56),
                 ('/data/deep_learning_study/cam_test/cat_dog.png', 243),
                 ('/data/deep_learning_study/cam_test/cat_dog.png',282),
                 ('/data/deep_learning_study/cam_test/pizza.jpg', 963))

    # Imagenet class
    imagenet_class = {56: 'king snake',243: 'bull mastiff', 282: 'tiger cat', 963: 'pizza'}

    
    cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], # 29번째 layer가 마지막 conv layer
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

    
    # Load Pretrained model
    model = VGGnet(make_layers(cfg['D'], batch_norm=False))
    pretrained_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16-397923af.pth')
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    model = model.to(DEVICE)
    
    ### DECLARE MODEL CLASS ###
    #여기에 ppt의 3가지 구현
    # Register_backward_hook, register_hook 를 이용하여 중간에 멈추고 gradient 뽑아오기
    
    class Grad_CAM:
        def __init__(self, model, target_layer):
            self.model = model
            self.target_layer = target_layer
            self.target_output = None
            self.target_grad = None
        
            def forward_hook(_, __, output):
                self.target_output = output 

            def backward_hook(_, __, grad_output):
                assert len(grad_output) == 1  # assert는 뒤의 조건이 True가 아니면 AssertError를 발생한다.
                
                self.target_grad = grad_output[0] # (1, 512, 14, 14)

            self.target_layer.register_forward_hook(forward_hook)  # input: 직전 layer의 출력, output: 현 layer의 출력
            self.target_layer.register_backward_hook(backward_hook) # grad_output: 모델 출력을 현 layer의 출력으로 미분한 grad.
                                                                    # grad_intput: grad_output * (현재layer의 출력을 현재 layer의 입력으로 미분한 grad) 
                                                                    # -> chain rule에 의해 = 모델 출력을 현 layer의 입력으로 미분한 grad 
        
        def forward(self, input):
            self.model.eval()
            self.model.zero_grad()  # optimizer.zero_grad() == model.zero_grad()
                                    # optimizer 1개, model 여러 개 -> optimizer.zero_grad() ///// optimizer 여러 개, model 1개 -> model.zero_grad()
            y, _ = self.model(input)
            
            return y
        
        def grad_cam(self, input, one_hot_target):
            #assert len(input.size()) == 3 ##
            #input = input.unsqueeze(0) 
            
            # one_hot_target: (1, 1000)
            # input: (1, 3, 224, 224)
            output = self.forward(input) # output: (1, 1000) 
            

            output.backward(one_hot_target)  # output이 scalar가 아닌 matrix이므로, 인자로 one_hot
            
            grad = self.target_grad #(1, 512, 14, 14)
            
            avg_pool = nn.AdaptiveAvgPool2d((14, 14))
            weights = avg_pool(grad) # (1,512,14,14)
            
            feature = self.target_output # (1, 512, 14, 14)
            
            
            feature = feature * weights # (1, 512, 14, 14)
            feature = torch.sum(feature, dim=1) # (1, 14, 14)
             
            feature = feature.squeeze()  # (14, 14)
            
            feature = F.relu(feature)
            
            
            feature = feature.cpu().detach().numpy()

            feature = cv2.resize(feature, (224,224), interpolation=cv2.INTER_CUBIC)  # scaling(확대)시 cv2.resize이용 -> (224, 224)
 
            return feature     
            
    
    # Guided Backporp: forward 뿐만 아니라 backward prop도 ReLU처럼 양수인것만 살고, 음수면 -> 0
    class Guided_Backprop:
        def __init__(self, model):
            self.model = model
        
            def backward_hook(_, grad_input, grad_output):
                assert len(grad_input)==1
                assert len(grad_output)==1
                y = ((grad_output[0]>0.0) * grad_input[0],) # grad_output[0]의 원소 중 양수인 것에 해당하는 grad_input[0]만 출력, 음수면 0
                        
                    
                
                return y

            for _, module_ in model.named_modules(): # named_modules(): 트리 구조의 모든 모듈(자신 포함)을 차례로 훑어본다.
                if isinstance(module_, nn.ReLU):
                    module_.register_backward_hook(backward_hook) # ReLU에서 사용
        
        def forward(self, input):
            self.model.eval()
            self.model.zero_grad()  # optimizer.zero_grad() == model.zero_grad()
                                    # optimizer 1개, model 여러 개 -> optimizer.zero_grad() ///// optimizer 여러 개, model 1개 -> model.zero_grad()
            y, _ = self.model(input)
            return y
            
        def get_guided_backprop(self, input, one_hot_target):
            #assert len(input.size()) == 3 ##
            #input = input.unsqueeze(0) # (1,3)
            if input.grad is not None:
                input.grad.zero_()
            input.requires_grad_(True) # input의 grad를 기록 시작
            output = self.forward(input) # (1,1000)
                  
            output.backward(one_hot_target)
            
            y = input.grad
            y = y.cpu().detach().numpy()
            
            return y
        
            
    for i in range(len(test_list)):
        img_path = test_list[i][0]
        target_class = test_list[i][1]
        

        one_hot_target = torch.zeros([1, NUM_CLASSES], dtype=torch.float) # (1,1000)
        one_hot_target[0][target_class] = 1 # 1000개 중에 특정 class를 대표하는 숫자의 자리만 1
        one_hot_target = one_hot_target.to(DEVICE)

        input_image = Image.open(img_path).convert('RGB')
        input_image = preprocess_image(input_image) # (1, 3, 224, 224)
        input_image = input_image.to(DEVICE)

        ### FORWARD MODEL FILE ###
    
        for _, module_ in model.named_modules(): 
            if isinstance(module_, nn.Conv2d):
                target_layer = module_
        
        
        # Grad_CAM
        
        cam = Grad_CAM(model, target_layer)
      
        grad_cam = cam.grad_cam(input_image, one_hot_target)
        
        one = np.ones((1, 3, 224, 224))
        out = grad_cam * one
        
        
        
        # Guided Backprop.
        GB = Guided_Backprop(model)
        guided_backprop = GB.get_guided_backprop(input_image, one_hot_target) 
        
        
        
        
        
        # Guided Grad_CAM
       
        GG_CAM = guided_backprop * out
        
        
        
        
        
        
        out = torch.Tensor(out)
        out = out.to(DEVICE)
        
        guided_backprop = torch.Tensor(guided_backprop)
        guided_backprop = guided_backprop.to(DEVICE)
        
        GG_CAM = torch.Tensor(GG_CAM)
        GG_CAM = GG_CAM.to(DEVICE)
        
        
        
        
        
        
        out = reprocess_image(out)
        out = cv2.applyColorMap(out, cv2.COLORMAP_JET)       
        plt.imsave('Grad_CAM_{}.png'.format(i+1), out)
                   
        
        guided_backprop = reprocess_image(guided_backprop)
        plt.imsave('GB_{}.png'.format(i+1), guided_backprop)
        
        
        GG_CAM = reprocess_image(GG_CAM) 
        plt.imsave('GG_CAM_{}.png'.format(i+1), GG_CAM)        
        
        input_image = reprocess_image(input_image)
        plt.imsave('original_{}.png'.format(i+1), input_image)
        
    
    
        
    
if __name__ == "__main__":
    main()