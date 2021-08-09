"""
Documentation at https://github.com/dmacneill/AxisID. 
"""

import torch
import torch.nn as nn

class Network(nn.Module):
    
    """Convolutional neural net for axis-angle prediction
    """
    
    def __init__(self):
        
        super().__init__()
        
        self.cnn_layers = nn.Sequential(
        nn.Conv2d(3,32, kernel_size = 3, stride = 1, padding = 0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(32,16, kernel_size = 4, stride = 1, padding = 0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16,16, kernel_size = 3, stride = 1, padding = 0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(16,16, kernel_size = 3, stride = 1, padding = 0),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=4, stride=4))

        self.linear_layers = nn.Sequential(
        nn.Dropout(p=0.5), nn.Linear(16*7*7, 2))
        
    def forward(self, x):
        
        x = x.float()
        maxs = x.max(dim = 3, keepdim = True)[0].max(dim = 2, keepdim = True)[0]
        mins = x.min(dim = 3, keepdim = True)[0].min(dim = 2, keepdim = True)[0]
        x = 2*(x-0.5*(maxs+mins))/(maxs-mins)
        
        x = self.cnn_layers(x)
        x = self.linear_layers(x.flatten(start_dim = 1))
        
        return x/torch.sqrt(torch.sum(x*x, axis = 1, keepdim = True))#The model outputs a unit vector
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)