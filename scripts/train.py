import os
import time
import datetime
import argparse
import yaml
import torch
import tensorboardX

from GPCR.dataset.constructor import construct_loader
from GPCR.model.constructor import construct_model
from GPCR.utils.engine import train_epoch, evaluate
from GPCR.utils.checkpoint import save_checkpoint

from GPCR.utils.global_var import step

def load_config(args):
    
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        
    return cfg

def parse_args():
    
    parser = argparse.ArgumentParser(
        description='GPCR pretraining script'
    )
    
    parser.add_argument(
        '--cfg',
        default='configs/default.yml',
        type=str,
        help='Config file path'
    )
    
    args = parser.parse_args()
    
    return args
    
def main():
    
    args = parse_args()
    cfg = load_config(args)
    
    global step
    
    if cfg['tensorboard']:
        train_writer = tensorboardX.SummaryWriter(
            os.path.join(
                cfg['output_dir'], 
                'logs',
                'train'
            )
        )
        test_writer = tensorboardX.SummaryWriter(
            os.path.join(
                cfg['output_dir'],
                'logs',
                'test',
            )
        )
    
    print(args)
    print(cfg)
    
    device = torch.device('cpu')
    if cfg['n_gpus'] > 0 and torch.cuda.device_count():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    
    print('Prepare data loaders')
    _, train_data_loader = construct_loader(cfg, 'train')
    _, val_data_loader = construct_loader(cfg, 'test')
    
    print('Prepare models')
    base_model, optimizer, lr_scheduler, start_epoch = construct_model(
        cfg, 'base_model', len(train_data_loader), finetune=True
    )
    
    print(base_model)
        
    if cfg['test_only']:
        print('Test only')
        evaluate(
            base_model, 
            val_data_loader, 
            device, 
            test_writer, 
            cfg
        )
        
        return
    
    print('Start training')
    start_time = time.time()
    for cur_epoch in range(start_epoch, cfg['epochs']):
        train_epoch(
            base_model, 
            optimizer, 
            lr_scheduler, 
            train_data_loader, 
            device, 
            cur_epoch, 
            train_writer, 
            cfg
        )
        evaluate(
            base_model, 
            val_data_loader, 
            device, 
            test_writer, 
            cfg
        )
        
        if cfg['output_dir']:
            save_checkpoint(cfg['output_dir'], base_model, optimizer, lr_scheduler, 
                            None, cur_epoch, step, args, cfg)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total training time for {} epochs: {}'.format(
        cfg['epochs'] - start_epoch,
        total_time_str
    ))

if __name__ == '__main__':
    
    main()
    