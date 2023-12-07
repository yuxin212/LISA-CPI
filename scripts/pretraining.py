import os
import sys
import time
import datetime
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorboardX

from GPCR.dataset.constructor import construct_loader
from GPCR.model.constructor import construct_model
from GPCR.model.losses import get_loss_func
from GPCR.model.optimizer import build_optimizer
from GPCR.utils.scheduler import WarmupMultiStepLR
from GPCR.utils.engine import train_epoch, evaluate
from GPCR.utils.checkpoint import save_checkpoint, load_checkpoint

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
        val_writer = tensorboardX.SummaryWriter(
            os.path.join(
                cfg['output_dir'],
                'logs',
                'val',
            )
        )
    
    print(args)
    print(cfg)
    
    device = torch.device('cpu')
    if cfg['n_gpus'] > 0 and cfg['n_gpus'] <= torch.cuda.device_count():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    
    print('Prepare data loaders')
    train_data_loader = construct_loader(cfg, 'train')
    val_data_loader = construct_loader(cfg, 'val')
    
    print('Prepare models')
    base_model, optimizer_base, lr_scheduler_base, start_epoch = construct_model(
        cfg, 'base_model', len(train_data_loader)
    )
    matcher, optimizer_matcher, lr_scheduler_matcher, start_epoch = construct_model(
        cfg, 'matcher', len(train_data_loader)
    )
    generator, optimizer_gen, lr_scheduler_gen, start_epoch = construct_model(
        cfg, 'gen', len(train_data_loader)
    )
    discriminator, optimizer_disc, lr_scheduler_disc, start_epoch = construct_model(
        cfg, 'disc', len(train_data_loader)
    )
    
    print(base_model)
    print(matcher)
    print(generator)
    print(discriminator)
    
    if cfg['test_only']:
        print('test only')
        evaluate()
        return
    
    print('Start training')
    start_time = time.time()
    for cur_epoch in range(start_epoch, cfg['epochs']):
        train_epoch(
            base_model, matcher, generator, discriminator, 
            optimizer_base, optimizer_matcher, optimizer_gen, optimizer_disc, 
            lr_scheduler_base, lr_scheduler_matcher, lr_scheduler_gen, lr_scheduler_disc, 
            train_data_loader, 
            device, 
            cur_epoch, 
            train_writer, 
            cfg
        )
        evaluate(
            base_model, 
            matcher, 
            generator, 
            discriminator, 
            val_data_loader, 
            device, 
            val_writer, 
            cfg
        )
        
        if cfg['output_dir']:
            save_checkpoint(cfg['output_dir'], base_model, optimizer_base, lr_scheduler_base, 
                            None, cur_epoch, step, args, cfg)
            save_checkpoint(cfg['output_dir'], matcher, optimizer_matcher, lr_scheduler_matcher, 
                            None, cur_epoch, step, args, cfg)
            save_checkpoint(cfg['output_dir'], generator, optimizer_gen, lr_scheduler_gen, 
                            None, cur_epoch, step, args, cfg)
            save_checkpoint(cfg['output_dir'], discriminator, optimizer_disc, lr_scheduler_disc, 
                            None, cur_epoch, step, args, cfg)
        
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total training time for {} epochs: {}'.format(
        cfg['epochs'] - start_epoch,
        total_time_str
    ))

if __name__ == '__main__':
    
    main()
    