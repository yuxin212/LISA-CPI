import os
import sys
import torch

def save_checkpoint(path, model, optimizer, lr_scheduler, amp, epoch,  
                    step, args, cfg, data_parallel=True):
    
    if not os.path.exists(path):
        print('Checkpoint save path not exist. Creating the save path')
        os.makedirs(path, exist_ok=True)
        
    model_name = model.module.name if data_parallel else model.name
    model = model.module if data_parallel else model
        
    checkpoint = {
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'lr_scheduler_state': lr_scheduler.state_dict(),
        'amp_state': amp.state_dict() if amp else None,
        'epoch': epoch,
        'step': step,
        'args': args,
        'cfg': cfg
    }
    
    torch.save(checkpoint, os.path.join(path, '{}_{}.pth'.format(model_name, epoch)))
    if model_name == 'BaseModel': 
        torch.save(checkpoint, os.path.join(path, 'ckpt.pth'))
    
def load_checkpoint(path, model, optimizer=None, lr_scheduler=None, 
                    amp=None, data_parallel=False, reset_epoch=False):
    
    if not os.path.exists(path):
        sys.exit('Checkpoint file not exist! Exiting...')
        
    model_state = model.module if data_parallel else model
    model_dict = model_state.state_dict()
    
    checkpoint = torch.load(path, map_location='cpu')
    
    pretrained_dict = checkpoint['model_state']
    matched_pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and v.size() == model_dict[k].size()
    }
    
    # layers without loaded pretrained weights
    plain_layers = [
        k 
        for k in model_dict.keys() 
        if k not in matched_pretrained_dict
    ]
    
    if plain_layers:
        for k in plain_layers:
            print('Layer {} is not loaded with pretrained weights'.format(k))
    
    # pretrained weights not loaded to the model
    wasted_weights = [
        k
        for k in pretrained_dict.keys()
        if k not in matched_pretrained_dict
    ]
    
    if wasted_weights:
        for k in wasted_weights:
            print('Pretrained weight {} is not used in the model'.format(k))
    
    model_state.load_state_dict(matched_pretrained_dict, strict=False)
    
    epoch = -1
    if 'epoch' in checkpoint.keys() and not reset_epoch:
        epoch = checkpoint['epoch']
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        if lr_scheduler:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state'])
        if amp:
            amp.load_state_dict(checkpoint['amp_state'])
            
    else:
        epoch = 0
        
    return epoch