import torch
import torch.nn as nn
import torch.nn.functional as F


class Alex(nn.Module):
        def __init__(self):
            super(Alex, self).__init__()
            
            self.Conv = nn.Sequential(
                # Conv1: input: -1x3x227x227
                nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),     # (227-11+0) / 4 + 1 = 55
                nn.ReLU(),
                nn.LocalResponseNorm(size=5, k=2),                                                  # default: alpha=0.0001, beta=0.75
                nn.MaxPool2d(kernel_size=3, stride=2),                                              # (55-3+1)/2 = 26.5 = 27 -> 27x27x96
                
                # Conv2
                # input: -1x27x27x96                                                                                     
                nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),    # (27-5+4) / 1 + 1 = 27
                nn.ReLU(),
                nn.LocalResponseNorm(size=5, k=2),                                                  # default: alpha=0.0001, beta=0.75
                nn.MaxPool2d(kernel_size=3, stride=2),                                              # (27-3+1)/2 = 12.5 = 13 -> 13x13x256
                                                                                                  
                # Conv3
                # input: -1x13x13x256
                nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),   # (13-3+2) / 1 + 1 = 13
                nn.ReLU(),                                                              # 13x13x384
                
                # Conv4
                # input: -1x13x13x384                                                                                    
                nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),   # (13-3+2) / 1 + 1 = 13
                nn.ReLU(),                                                                          # 13x13x384
                
                # Conv5
                # input: -1x13x13x384  
                nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),   # (13-3+2) / 1 + 1 = 13
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2)                                               # (13-3+1)/2 = 5.5 = 6 -> 6x6x256
                                                                                                   
            )
            
            
         
            self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
            
            self.FC = nn.Sequential(
                # FC1
                nn.Dropout(p=0.5),
                nn.Linear(6*6*256, 4096),   # 6*6*256 자리에 나중에 input shape이 변하면 -> 변수로 받는다.
                nn.ReLU(),
                
                # FC2
                nn.Dropout(p=0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                
                #FC3
                nn.Linear(4096, 64)
            )
        
        def init_bias_weight(self):
            for layer in self.Conv:
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                    nn.init.constant_(layer.bias, 0)
                nn.init.constant_(self.Conv[4].bias, 1)
                nn.init.constant_(self.Conv[10].bias, 1)
                nn.init.constant_(self.Conv[12].bias, 1)
            
        '''
            for layer in self.FC:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                    nn.init.constant_(layer.bias, 1)
        '''

        def forward(self, x):
            out = self.Conv(x)
            out = self.avgpool(out)
            out = out.view(-1, 256*6*6)

            out = self.FC(out)
            
            return out