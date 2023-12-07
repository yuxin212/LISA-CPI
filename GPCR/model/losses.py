import torch
import torch.nn as nn

__losses__ = {
    'BCELoss': nn.BCELoss,
    'L1Loss': nn.L1Loss,
    'MSELoss': nn.MSELoss, 
    'CrossEntropyLoss': nn.CrossEntropyLoss,
    'NLLLoss': nn.NLLLoss,
    'BCEWithLogitsLoss': nn.BCEWithLogitsLoss,
}

def get_loss_func(loss_name):
    
    assert (loss_name in __losses__), 'Not supported loss function! '
    
    return __losses__[loss_name]