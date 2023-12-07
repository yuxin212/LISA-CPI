import numpy as np
import torch
import torch.nn as nn
import torchvision

def get_base_cnn(model_name):
    
    if model_name == 'ResNet18':
        model = torchvision.models.resnet18(weights=None)
    elif model_name == 'ResNet34':
        model = torchvision.models.resnet34(weights=None)
    elif model_name == 'ResNet50':
        model = torchvision.models.resnet50(weights=None)
    elif model_name == 'ResNet101':
        model = torchvision.models.resnet101(weights=None)
    elif model_name == 'ResNet152':
        model = torchvision.models.resnet152(weights=None)
    else:
        raise NotImplementedError('{} is not supported'.format(model_name))
        
    features = model.fc.in_features
    
    return nn.Sequential(*list(model.children())[:-1]), features

def init_weights(m):
    
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0., np.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0., 0.02)
        m.bias.data.fill_(0.)
    