import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

class SimpleModel_Cifar(nn.Module):
    # Simple Model for Normalized LMS Training
    def __init__(self):
        super(SimpleModel_Cifar, self).__init__()
        self.conv1 = nn.Conv2d(3,8,3,padding=1,bias=False)
        self.bath1 = nn.BatchNorm2d(8)
        self.conv21 = nn.Conv2d(8,16,1,padding=0, stride=2, bias=False)
        self.bath21 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(8,16,3,padding=1, stride=2, bias=False)
        self.bath2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16,32,3,padding=1, stride=2, bias=False)
        self.bath3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32,64,3,padding=1, stride=2, bias=False)
        self.bath4 = nn.BatchNorm2d(64)
        self.drop1 = nn.Dropout(0.2)
        
        self.fc1   = nn.Linear(64*4*4, 10)
        
        
    def forward(self, x):
        
        
        x = self.conv1(x)
        x = self.bath1(x)
        x1 = self.conv21(x)
        x1 = self.bath21(x1)
        x = self.conv2(x)
        x = self.bath2(x)
        x = F.relu(x)
        x = x + x1
        x = self.conv3(x) 
        x = self.bath3(x)
        x = F.relu(x)
        x = self.conv4(x) 
        x = self.bath4(x)
        x = self.drop1(x)
        x = F.relu(x)
        
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        
        return x 
    
 