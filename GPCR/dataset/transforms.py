import torch
import torchvision.transforms as T

def minmax_normalize(data, dmin, dmax):
    
    data -= dmin
    data /= (dmax - dmin)
    
    return (data - 0.5) * 2

def denormalize(data, dmin, dmax):
    
    data /= 2
    data += 0.5
    data = data * (dmax - dmin) + dmin
    
    return data

def transforms_gpcr_data(
    img_size=224,
    hflip=0.5,
    vflip=0.0,
    gscale=0.2,
    rot_deg=360,
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
    sep=False
):
    
    if isinstance(img_size, tuple):
        img_size = img_size[-2:]
    else:
        img_size = img_size
        
    transform = []
    if img_size:
        transform = [
            T.CenterCrop(img_size),
        ]
    
    if hflip > 0.0:
        transform += [T.RandomHorizontalFlip(p=hflip)]
    if vflip > 0.0:
        transform += [T.RandomVerticalFlip(p=vflip)]
    if gscale > 0.0:
        transform += [T.RandomGrayscale(p=gscale)]
    if rot_deg > 0.0:
        transform += [T.RandomRotation(degrees=rot_deg)]
        
    transform += [T.ToTensor()]
    
    norm_tf = [
        T.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
    ]
    
    if sep:
        return T.Compose(transform), T.Compose(norm_tf)
    else:
        return T.Compose(transform + norm_tf)

class MinMaxNormalize(object):
    
    def __init__(self, dmin, dmax):
        
        self.datamin = dmin
        self.datamax = dmax
        
    def __call__(self, data):
        
        return minmax_normalize(data, self.datamin, self.datamax)
    
