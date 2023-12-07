import torch
import torch.nn as nn
from GPCR.model.model import BaseModel
from GPCR.model.optimizer import build_optimizer
from GPCR.utils.checkpoint import load_checkpoint
from GPCR.utils.scheduler import WarmupMultiStepLR

def construct_model(cfg, name, data_size, finetune=False, prediction=False):
    
    assert (name in ['base_model', 'matcher', 'gen', 'disc']), 'Not supported model {}!'.format(name)
    
    if name == 'base_model':
        model = BaseModel(cfg)
    
    if prediction:
        load_checkpoint(cfg['resume'], model)
        optimizer, lr_scheduler, start_epoch = None, None, None
        
    else:
        optimizer = build_optimizer(model, cfg)
    
        warmup_iters = cfg['scheduler']['lr_warmup_epochs'] * data_size
        lr_milestones = [data_size * m for m in cfg['scheduler']['lr_milestones']]
        lr_scheduler = WarmupMultiStepLR(
            optimizer,
            milestones=lr_milestones,
            gamma=cfg['scheduler']['lr_gamma'],
            warmup_iters=warmup_iters,
            warmup_factor=1e-5
        )
    
        if cfg['resume']:
            if finetune:
                start_epoch = load_checkpoint(cfg['resume'], model, 
                                              reset_epoch=True)
            else:
                start_epoch = load_checkpoint(cfg['resume'], model, 
                                              optimizer, lr_scheduler)
        else:
            start_epoch = 0
    
    if cfg['n_gpus'] > 0 and torch.cuda.device_count():
        device0 = torch.device(cfg['gpu_ids'][0])
        if cfg['n_gpus'] > 1:
            model = nn.DataParallel(model, device_ids=cfg['gpu_ids'])
            print('Using {} GPUs...'.format(cfg['n_gpus']))
        else:
            print('Using single GPU...')
    else:
        print('Using CPU...')
        device0 = torch.device('cpu')
    
    model.to(device0)
        
    return model, optimizer, lr_scheduler, start_epoch
