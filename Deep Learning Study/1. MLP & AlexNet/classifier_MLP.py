import torch
import numpy as np

from torchvision import datasets
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"



num_workers = 0
batch_size = 20

transform = transforms.ToTensor()

train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)
# root를 수정: /data/deep_learning_study/ -> download 할 필요x (PPT 참고)


train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)


batch_iterator = iter(train_loader)
images = next(batch_iterator)
print(images[0].shape)


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
        x = self.fc4(x)
        return x

model2 = Net2()
if torch.cuda.is_available():
    model2.cuda()     # model2를 GPU에 넣어줌
    

criterion = nn.CrossEntropyLoss()
optimizer_adam = torch.optim.Adam(model2.parameters(), lr=1e-3)


# Train
n_epochs = 10
model2.train()

for epoch in range(n_epochs):
    train_loss = 0.0
    for data, target in train_loader:
        if torch.cuda.is_available():
            data = data.cuda()         # data를 GPU에 넣어줌
            target = target.cuda()     # target을 GPU에 넣어줌
    
        optimizer_adam.zero_grad()
        
        output = model2(data)
        
        loss = criterion(output, target)
        loss.backward()
        
        optimizer_adam.step()
        
        train_loss += loss.item()*data.size(0)
        
    train_loss = train_loss / len(train_loader.dataset)
    
    print("Epoch: {} \tTraining Loss: {:.6f}".format(epoch+1, train_loss))


    
# Test
test_loss = 0.0
class_correct = [0. for i in range(10)]
class_total = [0. for i in range(10)]

model2.eval()

for data, target in test_loader:
   
    if torch.cuda.is_available():
        data = data.cuda()         # data를 GPU에 넣어줌
        target = target.cuda()     # target을 GPU에 넣어줌
    
    output = model2(data)
    
    loss = criterion(output, target)
    
    test_loss += loss.item()*data.size(0)
    
    _, pred = torch.max(output,1)
    
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    
    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

test_loss = test_loss/len(test_loader.dataset)
print('Test loss: {:.6f}'.format(test_loss))

for i in range(10):
    if class_total[i]>0:
        print('Test Accuracy of %5s: %2d%% (%2d/%2d)' %(str(i), 100*class_correct[i]/class_total[i], class_correct[i], class_total[i]))
    else:
        print('Test Accuracy of %5s: N/A' %(str(i)))

print('\nTest Accuracy (overall): %2d%% (%2d/%2d)' %(100*np.sum(class_correct)/np.sum(class_total), np.sum(class_correct), np.sum(class_total)))

torch.save(model2, "/deep_learning_study/week6") 