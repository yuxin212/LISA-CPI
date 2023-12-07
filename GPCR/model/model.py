import torch
import torch.nn as nn
from .utils import get_base_cnn, init_weights
from .modules import Conv2dBlock, DeConv2dBlock, LinearBlock

__basecnn__ = ['ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152']
__activation__ = ['LeakyReLU', 'ReLU', 'Sigmoid', 'Tanh']

class BaseModel(nn.Module):
    def __init__(self, cfg):
        
        super(BaseModel, self).__init__()
        
        self.cfg = cfg['model']['base_model']
        self.name = self.cfg['name']
        self.model_name = self.cfg['model']
        self.target_classes = self.cfg['target_classes']
        if 'final_activation' in self.cfg.keys():
            self.final_activation = self.cfg['final_activation']
        else:
            self.final_activation = None
        if 'rep' in self.cfg.keys():
            self.rep = self.cfg['rep']
        else:
            self.rep = None
        
        assert (self.model_name in __basecnn__), 'Not supported base cnn model! '
        assert (self.final_activation in __activation__), 'Not supported final activation layer'
        
        self.model, self.num_features = get_base_cnn(self.model_name)
        self.model.apply(init_weights)
        
        if self.final_activation == 'LeakyReLU':
            self.final_act_layer = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.final_act_layer = nn.Identity()
        
        if self.rep:
            self.target_classifier = nn.Linear(
                in_features=self.num_features + self.rep['in_feats'][-1], 
                out_features=self.target_classes
            )
        else:
            self.target_classifier = nn.Linear(
                in_features=self.num_features, 
                out_features=self.target_classes
            )
    
    def forward(self, x, rep):
        
        feat = self.model(x)
        feat = feat.view(feat.size(0), -1)

        if rep is not None:
            feat = torch.cat((feat, rep), axis=-1)
        
        output = self.target_classifier(feat)
        output = self.final_act_layer(output)
        
        return feat, output
