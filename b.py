
"""
Yingyi Luo
ECE559 hm6
To test the modle, just use anaconda promt , and print cd to the path.
Use 0603-675567577-Luo.py img_path, for example： python 0603-675567577-Luo.py image1.png
"""
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import Image

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

if __name__=='__main__':
    if(len(sys.argv)==2):
        img_path = sys.argv[1]
    else:
        print("Format is wrong! Please use 0603-675567577-Luo.py img_path, for example, python3 0603-675567577-Luo.py image1.png")
        sys.exit(1)
    
    # img_path = 'Star_000cf8ca-2a88-11ea-8123-8363a7ec19e6.png'
    TRANSFORM_IMG = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        # transforms.Resize(100),
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),  # preposessing
    ])
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
    
   # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    model = torch.load("0602-674467577-Luo.pth").to(device)
    image = Image.open(img_path)
    x = TRANSFORM_IMG(image)
    x.unsqueeze_(0)  #[300 300 3] add to [1 200 200 3]
    #print(x.shape)

    output = model(x.to(device))
    y = output.argmax(1)
    print(classes[y])