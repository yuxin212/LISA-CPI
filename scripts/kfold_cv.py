import os
import time
import datetime
import argparse
import yaml
import torch
import tensorboardX
from sklearn.model_selection import KFold

from GPCR.dataset.constructor import construct_loader, construct_kfold_loader
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
                'test'
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
    dataset, _ = construct_loader(cfg, 'train')
    __, test_data_loader = construct_loader(cfg, 'test')
    
    kfold = KFold(
        n_splits=cfg['kfold_splits'],
        random_state=cfg['rng_seed'],
        shuffle=True
    )
    
    print('Prepare models')
    base_model, optimizer, lr_scheduler, start_epoch = construct_model(
        cfg, 'base_model', len(_) * (cfg['kfold_splits'] - 1) / cfg['kfold_splits'], finetune=True
    )
    
    print(base_model)
        
    if cfg['test_only']:
        print('Test only')
        evaluate(
            base_model, 
            test_data_loader, 
            device, 
            test_writer, 
            cfg
        )
        
        return
    
    print('Start training')
    for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
        print('Fold {}/{}'.format(fold + 1, cfg['kfold_splits']))
        
        train_data_loader, val_data_loader = construct_kfold_loader(cfg, dataset, train_ids, val_ids)
        
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
                val_writer, 
                cfg,
                header='Validation: '
            )
            evaluate(
                base_model, 
                test_data_loader, 
                device, 
                test_writer,
                cfg,
                header='Test: '
            )
        
            if cfg['output_dir']:
                output_dir = os.path.join(cfg['output_dir'], 'fold_{}'.format(fold))
                save_checkpoint(output_dir, base_model, optimizer, lr_scheduler, 
                                None, cur_epoch, step, args, cfg)
        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Total training time for {} epochs: {}'.format(
            cfg['epochs'] - start_epoch,
            total_time_str
        ))
    

if __name__ == '__main__':
    
    main()
    