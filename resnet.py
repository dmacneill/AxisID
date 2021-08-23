"""
Documentation at https://github.com/dmacneill/AxisID. 
"""

import torch
import torch.nn as nn
import torchvision

class Identity(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Network(nn.Module):
    
    """Convolutional neural net for axis-angle prediction
    """
    
    def __init__(self):
        
        super().__init__()
        
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.layer3 = Identity()
        self.resnet.layer4 = Identity()
        self.resnet.fc = Identity()
        self.linear_layers = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(128,2))
        self.input_transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def forward(self, x):
        
        x = x/255.0
        x = self.input_transform(x)
        x = self.resnet(x)
        x = self.linear_layers(x)
        
        return x/torch.sqrt(torch.sum(x*x, axis = 1, keepdim = True))#The model outputs a unit vector
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)