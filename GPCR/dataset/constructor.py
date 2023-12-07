import torch
import torchvision.transforms as T
from torch.utils.data.sampler import RandomSampler, SubsetRandomSampler
from torch.utils.data._utils.collate import default_collate
from GPCR.dataset.GPCR import GPCR, GPCRPrediction
from GPCR.dataset.transforms import transforms_gpcr_data, MinMaxNormalize

def construct_loader(cfg, split):
    
    assert split in ['train', 'val', 'test']
    
    if split == 'train':
        batch_size = cfg['train']['batch_size']
        drop_last = True
        anno = cfg['train']['anno']
    elif split == 'val':
        batch_size = cfg['val']['batch_size']
        drop_last = False
        anno = cfg['val']['anno']
    elif split == 'test':
        batch_size = cfg['test']['batch_size']
        drop_last = False
        anno = cfg['test']['anno']
        
    transform_data, normalize = transforms_gpcr_data(
        img_size=cfg['transforms']['img_size'],
        hflip=cfg['transforms']['hflip'],
        vflip=cfg['transforms']['vflip'],
        gscale=cfg['transforms']['gscale'],
        rot_deg=cfg['transforms']['rot_deg'],
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        sep=True
    )
        
    transform_label = T.Compose([
        MinMaxNormalize(cfg['label']['min'], cfg['label']['max']), 
    ])
    dataset = GPCR(anno, cfg, normalize, transform_data, transform_label)
        
    sampler = RandomSampler(dataset)
        
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        sampler=sampler, 
        num_workers=cfg['workers'],
        pin_memory=True, drop_last=drop_last,
        collate_fn=default_collate
    )
    
    return dataset, loader

def construct_kfold_loader(cfg, dataset, train_ids, test_ids):
    
    batch_size = cfg['train']['batch_size']
    num_workers = cfg['workers']
    
    train_subsampler = SubsetRandomSampler(train_ids)
    test_subsampler = SubsetRandomSampler(test_ids)
    
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, 
        sampler=train_subsampler, 
        num_workers=num_workers,
        pin_memory=True, drop_last=True,
        collate_fn=default_collate
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, 
        sampler=test_subsampler,
        num_workers=num_workers,
        pin_memory=True, drop_last=False, 
        collate_fn=default_collate
    )
    
    return train_loader, test_loader

def construct_prediction_loader(cfg, data_dir, rep_path):
    
    transform_data, normalize = transforms_gpcr_data(
        img_size=cfg['transforms']['img_size'],
        hflip=cfg['transforms']['hflip'],
        vflip=cfg['transforms']['vflip'],
        gscale=cfg['transforms']['gscale'],
        rot_deg=cfg['transforms']['rot_deg'],
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        sep=True
    )
    
    dataset = GPCRPrediction(data_dir, rep_path, cfg, normalize, transform_data)
    sampler = RandomSampler(dataset)
    
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg['pred']['batch_size'],
        sampler=sampler, 
        num_workers=cfg['workers'],
        pin_memory=True, drop_last=False,
        collate_fn=default_collate
    )
    
    return dataset, loader