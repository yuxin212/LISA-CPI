import torch.nn as nn
import torch.nn.functional as F

class Conv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3),
                 stride=1, padding=1, activation='LeakyReLU', norm='BatchNorm', 
                 inplace_activation=False):
        
        if activation:
            assert activation in __activations__
        
        bias = True
        if norm:
            assert norm in __norms__
            bias = False
            
        super(Conv2dBlock, self).__init__()
        
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              bias=bias)
        
        self.norm = None
        if norm == 'BatchNorm':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'InstanceNorm':
            self.norm = nn.InstanceNorm2d(out_channels)
        
        self.activation = None
        if activation == 'ReLU':
            self.activation = nn.ReLU(inplace_activation)
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(0.2, inplace=inplace_activation)
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
            
    def forward(self, x):
        
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        
        return x
    
class DeConv2dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3),
                 stride=1, padding=1, activation='LeakyReLU', norm='BatchNorm', 
                 inplace_activation=False):
        
        if activation:
            assert activation in __activations__
        
        bias = True
        if norm:
            assert norm in __norms__
            bias = False
            
        super(DeConv2dBlock, self).__init__()
        
        self.deconv = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size, stride=stride, padding=padding,
                                         bias=bias)
        
        self.norm = None
        if norm == 'BatchNorm':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'InstanceNorm':
            self.norm = nn.InstanceNorm2d(out_channels)
        
        self.activation = None
        if activation == 'ReLU':
            self.activation = nn.ReLU(inplace_activation)
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(0.2, inplace=inplace_activation)
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
            
    def forward(self, x):
        
        x = self.deconv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        
        return x
    
class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, activation='LeakyReLU', norm='BatchNorm',
                 inplace_activation=False):
        
        if activation:
            assert activation in __activations__
        
        bias = True
        if norm:
            assert norm in __norms__
            bias = False
            
        super(LinearBlock, self).__init__()
        
        self.linear = nn.Linear(in_features=in_features, out_features=out_features, bias=bias)
        
        self.norm = None
        if norm == 'BatchNorm':
            self.norm = nn.BatchNorm1d(out_features)
        elif norm == 'InstanceNorm':
            self.norm = nn.InstanceNorm1d(out_features)
        
        self.activation = None
        if activation == 'ReLU':
            self.activation = nn.ReLU(inplace_activation)
        elif activation == 'LeakyReLU':
            self.activation = nn.LeakyReLU(0.2, inplace=inplace_activation)
        elif activation == 'Tanh':
            self.activation = nn.Tanh()
        elif activation == 'Sigmoid':
            self.activation = nn.Sigmoid()
            
    def forward(self, x):
        
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        
        return x
    
__activations__ = ['ReLU', 'LeakyReLU', 'Tanh', 'Sigmoid']
__norms__ = ['BatchNorm', 'InstanceNorm']