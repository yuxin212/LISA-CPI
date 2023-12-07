import os
import json
import time
import pickle
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error

from GPCR.utils.logging import MetricLogger, SmoothedValue
from GPCR.model.losses import get_loss_func
from GPCR.dataset.transforms import denormalize

from GPCR.utils.global_var import step

def train_base(
    model, 
    img, 
    label, 
    rep, 
    cfg
):
    
    criterion = get_loss_func(cfg['losses']['base_model'])()
    
    model.train()
    _, output = model(img, rep)
    loss = criterion(output.squeeze(), label)
    
    return loss

def train_epoch(
    model, 
    optimizer, 
    lr_scheduler, 
    train_loader, 
    device, 
    cur_epoch, 
    writer, 
    cfg
):
    
    global step
    
    logger = MetricLogger(delimiter=' ')
    logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value}'))
    logger.add_meter('samples/s', SmoothedValue(window_size=1, fmt='{value}'))
    
    header = 'Epoch [{}]'.format(cur_epoch)
    itr = 0
    max_itr = len(train_loader)
    
    for img, label, rep, idx, gpcr in logger.log_every(
        train_loader, cfg['print_freq'], header
    ):
        start_time = time.time()
        
        img, label, rep = img.to(device), label.float().to(device), rep.to(device)
        
        loss = train_base(
            model, 
            img, 
            label, 
            rep, 
            cfg
        )
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        batch_size = img.shape[0]
        logger.update(
            loss=loss.item(), 
            lr=optimizer.param_groups[0]['lr']
        )
        logger.meters['samples/s'].update(batch_size / (time.time() - start_time))
        
        if writer:
            writer.add_scalar('loss', loss.item(), step[0])
        
        step[0] += 1
        itr += 1
        lr_scheduler.step()

def evaluate(
    model, 
    val_loader, 
    device, 
    writer, 
    cfg, 
    header='Test: '
):
    
    model.eval()

    
    logger = MetricLogger(delimiter=' ')
    header = header
    
    if 'eval_and_vis' in cfg.keys():
        criterion = get_loss_func(cfg['losses']['base_model'])(reduction='none')
        if cfg['eval_and_vis']['per_sample_result']:
            per_sample_result = [] # pd.DataFrame(columns=['name', 'prob', 'label', 'loss'])
        if cfg['eval_and_vis']['tsne']:
            latent_feat = []
    else:
        criterion = get_loss_func(cfg['losses']['base_model'])()
        per_sample_result = None
        latent_feat = None
    
    with torch.no_grad():
        sigmoid = nn.Sigmoid()
        
        for img, label, rep, idx, gpcr in logger.log_every(
            val_loader, cfg['print_freq'], header
        ):
            img = img.to(device, non_blocking=True)
            label = label.float().to(device, non_blocking=True)
            rep = rep.to(device, non_blocking=True)
            
            feat, output = model(img, rep)
            prob = sigmoid(output)
            loss = criterion(output.squeeze(), label)
            
            output_np = output.cpu().numpy().squeeze()
            label_np = label.cpu().numpy().squeeze()
            prob_np = prob.cpu().numpy().squeeze()
            
            if cfg['task_type'] == 'regression':
                rmse = np.sqrt(loss.mean().item())
                r = pearsonr(output_np, label_np).statistic
                mae = mean_absolute_error(output_np, label_np)
                mae = float(mae)
                logger.update(rmse=rmse)
                logger.update(r=r)
                logger.update(mae=mae)
                
            else:
                pred_label = np.where(prob_np > 0.5, 1, 0)
                accuracy = accuracy_score(label_np, pred_label)
                
                try:
                    auc = roc_auc_score(
                        label_np, 
                        prob_np)
                except:
                    auc = 0.
                logger.update(accuracy=accuracy)
                logger.update(auc=auc)
            
            logger.update(loss=loss.mean().item())
            
            
            if 'eval_and_vis' in cfg.keys():
                if cfg['eval_and_vis']['per_sample_result']:
                    for name, prob, output, label, gpcr_target, diff, feature in zip(
                        idx, prob_np, output_np, label_np, gpcr, loss.detach().cpu().numpy().tolist(), feat.detach().cpu().numpy()
                    ):
                        row = {
                            'name': name, 
                            'prob': prob.tolist(), 
                            'output': output.tolist(), 
                            'label': float(label), 
                            'gpcr': gpcr_target, 
                            'loss': float(diff),
                        }
                        
                        per_sample_result.append(row)
                        
                        if cfg['eval_and_vis']['tsne']:
                            per_sample_latent_feat = {
                                'drug': name,
                                'gpcr': gpcr_target,
                                'feature': feature,
                                'label': float(label)
                            }
                            latent_feat.append(
                                per_sample_latent_feat
                            )
            
    logger.synchronize_between_processes()
    
    if cfg['task_type'] == 'regression':
        print('loss: {:.8f} \n\
        rmse: {:.4f} \n\
        r: {:.6f} \n\
        mae: {:.4f}\n'.format(
            logger.loss.global_avg, 
            logger.rmse.global_avg, 
            logger.r.global_avg,
            logger.mae.global_avg
        ))
    else:
        print('loss: {:.8f} \n\
        accuracy: {:.4f} \n\
        auc: {:.6f} \n'.format(
            logger.loss.global_avg, 
            logger.accuracy.global_avg,
            logger.auc.global_avg
        ))
    
    if writer:
        writer.add_scalar('loss', logger.loss.global_avg, step[0])
        if cfg['task_type'] == 'regression':
            writer.add_scalar('rmse', logger.rmse.global_avg, step[0])
            writer.add_scalar('r', logger.r.global_avg, step[0])
            writer.add_scalar('mae', logger.mae.global_avg, step[0])
        else:
            writer.add_scalar('accuracy', logger.accuracy.global_avg, step[0])
            writer.add_scalar('auc', logger.auc.global_avg, step[0])
    
    if 'eval_and_vis' in cfg.keys():
        if cfg['eval_and_vis']['per_sample_result']:
            per_sample_path = os.path.join(cfg['output_dir'], 'per_sample_result.json')
            # per_sample_result.to_csv(per_sample_path)
            with open(per_sample_path, 'w') as f:
                json.dump(per_sample_result, f, indent=1)
        
        if latent_feat:
            # latent_feat = np.array(latent_feat)
            latent_feat_path = os.path.join(cfg['output_dir'], 'latent_feat.pkl')
            with open(latent_feat_path, 'wb') as f:
                pickle.dump(latent_feat, f, protocol=4)
            # np.save(latent_feat_path, latent_feat)

def make_prediction(
    model, 
    dataloader, 
    device,
    cfg, 
    out_dir, 
    header='Prediction '
):
    
    model.eval()
    
    logger = MetricLogger(delimiter=' ')
    header = header
    per_sample_result = []
    
    with torch.no_grad():
        
        for img, rep, drug_name, protein_name in logger.log_every(
            dataloader, cfg['print_freq'], header
        ):
            img = img.to(device, non_blocking=True)
            rep = rep.to(device, non_blocking=True)
            
            _, output = model(img, rep)
            output_np = output.cpu().numpy()
            
            output_np_denorm = denormalize(
                output_np, cfg['label']['min'], cfg['label']['max']
            )
            
            for drug, protein, pred in zip(drug_name, protein_name, output_np_denorm):
                result = {
                    'drug': drug,
                    'protein': protein,
                    'prediction': float(pred[0]),
                }
                per_sample_result.append(result)
            
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    with open(os.path.join(out_dir, 'per_sample_result.json'), 'w') as f:
        json.dump(per_sample_result, f, indent=1)