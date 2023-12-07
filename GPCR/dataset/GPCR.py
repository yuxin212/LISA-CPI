import os
import json
import torch
import numpy as np
from PIL import Image

class GPCR(torch.utils.data.Dataset):
    
    def __init__(self, anno, cfg, normalize=None, transform_data=None, transform_label=None):
        
        assert (os.path.exists(anno)), "Annotation file {} doesn't exist!".format(anno)
        
        self.cfg = cfg
        self.normalize = normalize
        self.transform_data = transform_data
        self.transform_label = transform_label
        
        with open(anno, 'r') as f:
            self.json_data = json.load(f)
    
    def __getitem__(self, index):
        
        item = self.json_data[index]
        data = item['data']
        label = item['label']
        if item['rep']:
            rep = np.load(item['rep'])
        else:
            rep = None
        fname = data
        gpcr = item['rep']
        
        if 'smooth' in self.cfg.keys():
            from scipy.ndimage import gaussian_filter1d
            rep = gaussian_filter1d(rep, self.cfg['smooth'])
        
        img = Image.open(fname).convert('RGB')
        
        if type(label) == type(list()):
            label = np.array(label).astype(np.float32)
        
        if self.transform_data:
            img = self.transform_data(img)
        if self.transform_label:
            label = self.transform_label(label)
        if self.normalize:
            img = self.normalize(img)
        
        return img, label, rep, fname, gpcr
    
    def __len__(self):
        
        return len(self.json_data)

class GPCRPrediction(torch.utils.data.Dataset):
    
    def __init__(self, data_dir, rep_path, cfg, normalize=None, transform_data=None):
        
        self.cfg = cfg
        self.normalize = normalize
        self.transform_data = transform_data
        
        if os.path.isdir(data_dir):
            self.data = [os.path.join(data_dir, data) for data in os.listdir(data_dir)]
        elif os.path.isfile(data_dir):
            self.data = [data_dir]
        
        self.rep = rep_path
        
    def __getitem__(self, index):
        
        drug_name = self.data[index]
        protein_name = self.rep
        
        img = Image.open(drug_name).convert('RGB')
        rep = np.load(self.rep)
        
        if 'smooth' in self.cfg.keys():
            from scipy.ndimage import gaussian_filter1d
            rep = gaussian_filter1d(rep, self.cfg['smooth'])
        
        rep = torch.from_numpy(rep)
        if self.transform_data:
            img = self.transform_data(img)
        if self.normalize:
            img = self.normalize(img)
        
        return img, rep, drug_name, protein_name
        
    def __len__(self):
        
        return len(self.data)
        
if __name__ == '__main__':
    pass