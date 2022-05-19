"""
hm6 ECE559 Yingyi luo
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pylab as plt
import torchvision.models as models
torch.manual_seed(0)
import random
random.seed(1)
import numpy as np
np.random.seed(2)
import os
import shutil
classes = [
  'Circle',
  'Square',
  'Octagon',
  'Heptagon',
  'Nonagon',
  'Star',
  'Hexagon',
  'Pentagon',
  'Triangle'
]

if not os.path.exists('train'):
    os.mkdir('train')
    for i in range(len(classes)):
        os.mkdir('train/' + str(i) + classes[i])

if not os.path.exists('test'):
    os.mkdir('test')
    for i in range(len(classes)):
        os.mkdir('test/' + str(i) + classes[i])

files = []
for r, d, f in os.walk('output'):
    for file in f:
        files.append(file)
files.sort()

for i in range(len(files)):
    file = files[i]
    ind = classes.index(file[:file.find('_')])
    if i % 10000 < 8000:
        shutil.copy('output/' + files[i], 'train/' + str(ind) + classes[ind] + '/' + files[i])
    else:
        shutil.copy('output/' + files[i], 'test/' + str(ind) + classes[ind] + '/' + files[i])



class CNN(nn.Module):
    def __init__(self, input_shape=(3, 200, 200)):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 7, 3, padding=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 16, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(3, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        n_size = self._get_conv_output(input_shape)
        print(n_size)
        self.fc1 = nn.Linear(n_size, 9)
        
        
    def _get_conv_output(self, shape):
        bs = 1
        input = torch.autograd.Variable(torch.rand(bs, *shape))
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = self.bn1(F.elu(self.conv1(x)))
        x = self.pool1(x)
        x = self.bn2(F.elu(self.conv2(x)))
        x = self.pool2(x)
        x = self.bn3(F.elu(self.conv3(x)))
        x = self.pool2(x)
        x = self.bn4(F.elu(self.conv4(x)))
        x = self.pool2(x)
        return x
        
    def forward(self, x):
        x = self.bn1(F.elu(self.conv1(x)))
        x = self.pool1(x)
        x = self.bn2(F.elu(self.conv2(x)))
        x = self.pool2(x)
        x = self.bn3(F.elu(self.conv3(x)))
        x = self.pool2(x)
        x = self.bn4(F.elu(self.conv4(x)))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)
        return x

def train_loop(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction and loss
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
    
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct
    

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    batch_size = 128
    
    
    TRAIN_DATA_PATH = "./train/"
    TEST_DATA_PATH = "./test/"
    TRANSFORM_IMG = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                #transforms.Resize(100),
                                transforms.ToTensor(),
                                transforms.Normalize((0,), (1,)), #preposessing

    ])

    train_data = torchvision.datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_data = torchvision.datasets.ImageFolder(root=TEST_DATA_PATH, transform=TRANSFORM_IMG)
    testloader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False) 

    
    classes = [
      'Circle',
      'Square',
      'Octagon',
      'Heptagon',
      'Nonagon',
      'Star',
      'Hexagon',
      'Pentagon',
      'Triangle'
    ]
    
    model = CNN().to(device)
    total_step = len(trainloader)
    learning_rate = 0.015#0.01
    

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()    #CrossEntropy交叉熵
    
    train_acc_hist = []
    test_acc_hist = []
    train_loss_hist = []
    test_loss_hist = []
    epochs = 50

    
    for t in range(epochs):
        print(f"Epoch {t+1}/{epochs}\n-------------------------------")
        train_loop(trainloader, model, loss_fn, optimizer)
        train_loss, train_acc = test_loop(trainloader, model, loss_fn)
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        test_loss, test_acc = test_loop(testloader, model, loss_fn)
        test_loss_hist.append(test_loss)
        test_acc_hist.append(test_acc)
        
    plt.figure()
    plt.ylabel("Loss (training and validation)")
    plt.xlabel("Training Steps")

    plt.plot(train_loss_hist, label='Training')
    plt.plot(test_loss_hist, label='Test')
    plt.legend()
    plt.grid()
    
    plt.figure()
    plt.ylabel("Accuracy (training and validation)")
    plt.xlabel("Training Steps")
    plt.ylim([0,1])
    plt.plot(train_acc_hist, label='Training')
    plt.plot(test_acc_hist, label='Test')
    plt.legend()
    plt.grid()
    plt.show()
    
    print("Done!")
    torch.save(model, 'model.pth')



    np.save('./test_loss_hist.npy', test_loss_hist)
    np.save('./test_acc_hist.npy', test_acc_hist)

    np.save('./train_loss_hist.npy', train_loss_hist)
    np.save('./train_acc_hist.npy', train_acc_hist)

'''
the network is learned from a paper named:2D geometric shapes dataset – for machine learning and pattern recognition, and the example given by the teacher troch3.py

cite: El Korchi A, Ghanou Y. 2D geometric shapes dataset–for machine learning and pattern recognition[J]. Data in Brief, 2020, 32: 106090.

'''