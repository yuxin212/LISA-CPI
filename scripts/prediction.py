import os
import yaml
import torch

from GPCR.dataset.constructor import construct_prediction_loader
from GPCR.model.constructor import construct_model
from GPCR.utils.engine import make_prediction

def load_config(args):
    
    with open(args.cfg, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    return cfg

def parse_args():
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='GPCR prediction script'
    )
    
    parser.add_argument(
        '--cfg', 
        default='configs/prediction/cfg.yml',
        type=str,
        help='config file path'
    )
    parser.add_argument(
        '--data-dir',
        default='data/pred/fda/imgs',
        type=str,
        help='molecular images path'
        'folder or single file'
    )
    parser.add_argument(
        '--rep-path',
        default='data/representations/pain/P08908.npy',
        type=str,
        help='intermediate representation file path'
    )
    parser.add_argument(
        '--out-dir',
        default='output/prediction',
        type=str,
        help='output path'
    )
    
    args = parser.parse_args()
    
    return args

def main():
    
    args = parse_args()
    cfg = load_config(args)
    
    print(args)
    print(cfg)
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    device = torch.device('cpu')
    if cfg['n_gpus'] > 0 and torch.cuda.device_count():
        device = torch.device('cuda')
    
    print('Loading data')
    _, dataloader = construct_prediction_loader(cfg, args.data_dir, args.rep_path)
    
    print('Prepare models')
    model, _, _, _ = construct_model(
        cfg, 'base_model', 0, prediction=True
    )
    
    print(model)
    
    print('Start prediction')
    make_prediction(
        model, 
        dataloader, 
        device, 
        cfg,
        args.out_dir
    )
    
    return

if __name__ == '__main__':
    
    main()