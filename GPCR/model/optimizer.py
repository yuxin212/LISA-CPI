import torch

__optimizers__ = ['Adam', 'SGD', 'AdamW']

def build_optimizer(model, cfg):
    
    model_name = model.name
    
    trainable_params = get_trainable_params(model)
    
    method = cfg['optimizers'][model_name]['method']
    assert (method in __optimizers__), 'Not supported optimizer method {}!'.format(method)
    
    if method == 'Adam':
        optimizer = torch.optim.Adam(
            trainable_params,
            lr=cfg['optimizers'][model_name]['lr'],
            betas=cfg['optimizers'][model_name]['betas'],
            weight_decay=cfg['optimizers'][model_name]['wd']
        )
    elif method == 'SGD':
        optimizer = torch.optim.SGD(
            trainable_params,
            lr=cfg['optimizers'][model_name]['lr'],
            momentum=cfg['optimizers'][model_name]['momentum'],
            weight_decay=cfg['optimizers'][model_name]['wd']
        )
    elif method == 'AdamW':
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=cfg['optimizers'][model_name]['lr'],
            betas=(0.9, 0.999),
            weight_decay=cfg['optimizers'][model_name]['wd']
        )
    
    return optimizer

def get_trainable_params(model):
    
    model_name = model.name
    
    grad_params = []
    no_grad_params = []
    
    for name, module in model.named_modules():
        for param in module.parameters(recurse=False):
            if param.requires_grad:
                grad_params.append(param)
            else:
                no_grad_params.append(param)

    trainable_params = [param for param in grad_params if len(param)]
    
    assert (
        len(grad_params) + len(no_grad_params) == len(list(model.parameters()))
    ), '{}: parameters size does not match: {} require grad, \
    {} does not require grad, {} total parameters'.format(
        model_name, 
        len(grad_params), 
        len(no_grad_params), 
        len(list(model.parameters()))
    )
    
    
    print('{}: {} require grad, {} does not require grad, {} total parametres'.format(
        model_name, 
        len(grad_params),
        len(no_grad_params),
        len(list(model.parameters()))
    ))
    
    return trainable_params
