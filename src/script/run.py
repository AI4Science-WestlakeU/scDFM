import accelerate
import torch
import torch.nn as nn
import tyro
from config.config_flow import FlowConfig as Config
import torch.nn.functional as F
import time
from torch.utils.data import Dataset, DataLoader
import jax
import random
from src.data_process.data import Data, PerturbationDataset
from src.flow_matching.ot import OTPlanSampler
from src.flow_matching.path import AffineProbPath
from src.flow_matching.solver import ODESolver
from src.models.instantiate_model import instantiate_model
from src.tokenizer.gene_tokenizer import GeneVocab
from src.models.perturbation.moduls import PerturbationEmbedding
import pdb
import tqdm
from src.flow_matching.path.scheduler import CondOTScheduler
import scanpy as sc
import os
from src.data_process.utils import build_generated_anndata

import json
from accelerate import Accelerator,DistributedDataParallelKwargs
import torchdiffeq
from tqdm import trange
import numpy as np
from cell_eval import MetricsEvaluator
import anndata as ad
import pandas as pd
from src.utils.utils import save_checkpoint, load_checkpoint, make_lognorm_poisson_noise, pick_eval_score, process_vocab, set_requires_grad_for_p_only, get_perturbation_emb

ot_sampler = OTPlanSampler(method="exact") 
path = AffineProbPath(scheduler=CondOTScheduler())

def gaussian_kernel(x, y, sigma=1.0):
    beta = 1.0 / (2.0 * sigma**2)
    dist = torch.cdist(x, y, p=2) ** 2
    return torch.exp(-beta * dist)

def mmd_loss(pred, tgt, sigma=1.0):
    xx = gaussian_kernel(pred, pred, sigma).mean(dim=(1))
    yy = gaussian_kernel(tgt, tgt, sigma).mean(dim=(1))
    xy = gaussian_kernel(pred, tgt, sigma).mean(dim=(1))
    return (xx + yy - 2 * xy).mean()

def pairwise_sq_dists(X, Y):
    # X:[m,d], Y:[n,d] -> [m,n]
    return torch.cdist(X, Y, p=2)**2

@torch.no_grad()
def median_sigmas(X, scales=(0.5, 1.0, 2.0, 4.0)):
    Z = X
    D2 = pairwise_sq_dists(Z, Z)
    tri = D2[~torch.eye(D2.size(0), dtype=bool, device=D2.device)]
    m = torch.median(tri).clamp_min(1e-12)          # 中位数的“距离平方”
    s2 = torch.tensor(scales, device=Z.device) * m   # 生成多尺度 σ^2
    sigmas = torch.sqrt(s2)                          # 转回 σ
    return [float(s.item()) for s in sigmas]

def mmd2_unbiased_multi_sigma(X, Y, sigmas):
    """
    """
    m, n = X.size(0), Y.size(0)
    Dxx = pairwise_sq_dists(X, X)   # [m,m]
    Dyy = pairwise_sq_dists(Y, Y)   # [n,n]
    Dxy = pairwise_sq_dists(X, Y)   # [m,n]

    vals = []
    for sigma in sigmas:
        beta = 1.0 / (2.0 * (sigma ** 2) + 1e-12)
        Kxx = torch.exp(-beta * Dxx)
        Kyy = torch.exp(-beta * Dyy)
        Kxy = torch.exp(-beta * Dxy)

        # 无偏（去对角）
        term_xx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1) + 1e-12)
        term_yy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1) + 1e-12)
        term_xy = Kxy.mean()  # / (m*n)
        vals.append(term_xx + term_yy - 2.0 * term_xy)

    return torch.stack(vals).mean()

def train_step(source, target, perturbation_id, vf, criterion, accelerator, noise_type='Poisson', mode="predict_y"):
    B = source.shape[0]
    device = accelerator.device
    
    input_gene_ids = torch.randperm(source.shape[-1], device=device)[:config.infer_top_gene]
    source = source[:,input_gene_ids]
    target = target[:,input_gene_ids]
    gene = gene_ids.repeat(B,1).to(device)
    gene_input = gene[:,input_gene_ids]
    
    if mode=="predict_y":
        # source, target = ot_sampler.sample_plan(source, target)
        t = torch.rand(B, device=device)
        if noise_type=="Gaussian":
            target_noise = torch.randn_like(source)
        elif noise_type=="Poisson":
            target_noise = make_lognorm_poisson_noise(
                target_log=source,
                alpha=getattr(config, "poisson_alpha", 0.8),           
                per_cell_L=getattr(config, "poisson_target_sum", 1e4),  # e.g., 1e4 or None
            )
        path_x1 = path.sample(t=t, x_0=target_noise, x_1=target)
        predicted_x_t_velocity = vf(gene_input,path_x1.x_t, path_x1.t,source,perturbation_id, gene_input, mode=mode)
        loss = ((predicted_x_t_velocity - path_x1.dx_t)**2).mean()
        
        if config.use_mmd_loss:
            x1_hat = path_x1.x_t + predicted_x_t_velocity*(1-t).unsqueeze(-1)
            sigmas = median_sigmas(target, scales=(0.5,1.0,2.0,4.0))
            
            _mmd_loss = mmd2_unbiased_multi_sigma(x1_hat, target, sigmas)
            # _mmd_loss = mmd_loss(x1_hat, target)
            loss = loss + _mmd_loss * config.gamma

    elif mode=="predict_p":
        t_p = torch.ones(B, device=device)  # Or uniform(0.7,1.0)
        predicted_p_embed = vf(gene_input, target, t_p, source, perturbation_id, gene_input, mode=mode)
        if hasattr(vf, "module"):
            base_vf = vf.module
        else:
            base_vf = vf
        p_embed_gt = base_vf.get_perturbation_emb(perturbation_id=perturbation_id, cell_1=source)
        pred = F.normalize(predicted_p_embed, dim=-1)
        tgt  = F.normalize(p_embed_gt.detach(), dim=-1)
        loss = 1 - (pred * tgt).sum(dim=-1).mean()  # cosine distance
    
    return loss

@torch.inference_mode()
def test(data_sampler, vf, accelerator,  batch_size=128, path='./',vocab=None,scheme='mse'):
    gene_ids_test = vocab.encode(list(data_sampler.adata.var_names))
    
    gene_ids_test = torch.tensor(gene_ids_test, dtype=torch.long, device=device)
    perturbation_name_list = data_sampler._perturbation_covariates
    control_data = data_sampler.get_control_data()
    all_pred_expressions = [control_data['src_cell_data']]
    obs_perturbation_name_pred = ['control']*control_data['src_cell_data'].shape[0]
    all_target_expressions = [control_data['src_cell_data']]
    obs_perturbation_name_real = ['control']*control_data['src_cell_data'].shape[0]
    count = 0
    print('perturbation_name_list:',len(perturbation_name_list))
    for perturbation_name in perturbation_name_list:
        perturbation_data = data_sampler.get_perturbation_data(perturbation_name)
        target = perturbation_data['tgt_cell_data']
        perturbation_id = perturbation_data['condition_id']
        source = control_data['src_cell_data']
        source = source.to(device)
        perturbation_id = perturbation_id.to(device)
        if config.perturbation_function == 'crisper':
            perturbation_name_crisper = [inverse_dict[int(p_id)] for p_id in perturbation_id[0].cpu().numpy()]
            perturbation_id = torch.tensor(vocab.encode(perturbation_name_crisper), dtype=torch.long, device=device)
            perturbation_id = perturbation_id.repeat(source.shape[0],1)
        
        idx = torch.randperm(source.shape[0])
        source = source[idx]
        N = 128
        source = source[:N]
        
        pred_expressions = []
        for i in trange(0, N, batch_size):
            batch_perturbation_id = perturbation_id[0].repeat(source[i:i+batch_size].shape[0],1)
            
            batch_perturbation_id = batch_perturbation_id.to(accelerator.device)
            
            pred_expression = generate_sample(wrapped_vf,source[i:i+batch_size],batch_perturbation_id,vf,gene_ids=gene_ids_test,gene_all=gene_ids_test)
            pred_expressions.append(pred_expression)
            
        pred_expressions = torch.cat(pred_expressions, dim=0).cpu().numpy()
        all_pred_expressions.append(pred_expressions)
        all_target_expressions.append(target)
        obs_perturbation_name_pred.extend([perturbation_name] * pred_expressions.shape[0])
        obs_perturbation_name_real.extend([perturbation_name] * target.shape[0])
        # count += 1
        # if count > 3:
        #     break

    all_pred_expressions = np.concatenate(all_pred_expressions, axis=0)
    all_target_expressions = np.concatenate(all_target_expressions, axis=0)
    obs_pred = pd.DataFrame({'perturbation':obs_perturbation_name_pred})
    obs_real = pd.DataFrame({'perturbation':obs_perturbation_name_real})
    pred = ad.AnnData(X=all_pred_expressions, obs=obs_pred)
    real = ad.AnnData(X=all_target_expressions, obs=obs_real)
    

    eval_score = None
    if accelerator.is_main_process:
        evaluator = MetricsEvaluator(
            adata_pred=pred,
            adata_real=real,
            control_pert="control",
            pert_col="perturbation",
            num_threads=32,
        )
        (results, agg_results) = evaluator.compute()
        
        results.write_csv(os.path.join(path, 'results.csv'))
        agg_results.write_csv(os.path.join(path, 'agg_results.csv'))
        pred.write_h5ad(os.path.join(path, 'pred.h5ad'))
        real.write_h5ad(os.path.join(path, 'real.h5ad'))

        eval_score = pick_eval_score(agg_results, scheme)
        print(f"Current evaluation score: {eval_score:.4f}")
    
    return eval_score

def wrapped_vf(target,t,source,perturbation_id,vf,gene_ids, gene_all):
    
    gene = gene_ids.repeat(source.shape[0],1).to(device)
    predicted_x_t_velocity = vf(gene,target,t,source,perturbation_id,gene_all)
    
    return predicted_x_t_velocity

@torch.no_grad()
def generate_sample(wrapped_vf,source,condition_vec=None,vf=None,gene_ids=None,gene_all=None,steps=20,method="rk4"):
    
    noise_type = config.noise_type
    if noise_type=="Gaussian":
        target_noise = torch.randn(source.shape[0],config.infer_top_gene,device=source.device)
    elif noise_type=="Poisson":
        target_noise = make_lognorm_poisson_noise(
            target_log=source,
            alpha=getattr(config, "poisson_alpha", 0.8),           
            per_cell_L=getattr(config, "poisson_target_sum", 1e4), 
        )
        
    traj = torchdiffeq.odeint(lambda t,x: wrapped_vf(x,t,source,condition_vec,vf,gene_ids,gene_all),
                              target_noise,
                              torch.linspace(0,1,steps).to(source.device),
                              atol=1e-4,
                              rtol=1e-4,
                              method=method)
    # t = torch.linspace(0,1,steps).to(source.device)
    # traj = [target_noise + 0.8*wrapped_vf(target_noise,t,source,condition_vec,vf,gene_ids,gene_all)]
    
    return torch.clamp(traj[-1], min=0)
    
if __name__ == "__main__":
    config = tyro.cli(Config)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs]
    )
    if accelerator.is_main_process:
        print(config)
        save_path = config.make_path()
        os.makedirs(save_path, exist_ok=True)
    device = accelerator.device
    
    data_manager = Data('./data')

    data_manager.load_data(config.data_name)
    data_manager.process_data(n_top_genes=config.n_top_genes, split_method=config.split_method, fold=config.fold, use_negative_edge=config.use_negative_edge, k=config.topk)
    train_sampler, valid_sampler, test_dl = data_manager.load_flow_data(batch_size=config.batch_size)
    
    train_dataset = PerturbationDataset(train_sampler, config.batch_size)
    dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False,num_workers=8,pin_memory=True,persistent_workers=True)  # batch_size=1 因为每个getitem本身就是一个batch
    if config.use_negative_edge:
        mask_path = os.path.join(data_manager.data_path, data_manager.data_name,'mask_fold_'+str(config.fold)+'topk_'+str(config.topk)+config.split_method+'_negative_edge'+'.pt')
    else:
        mask_path = os.path.join(data_manager.data_path, data_manager.data_name,'mask_fold_'+str(config.fold)+'topk_'+str(config.topk)+config.split_method+'.pt')
    vf = instantiate_model(config.model_type,
                           ntoken = config.ntoken,
                           d_model = config.d_model,
                           d_perturbation = config.d_model,
                           fusion_method = config.fusion_method,
                           perturbation_function = config.perturbation_function,
                           mask_path = mask_path
                           )
    
    model_path = config.make_path()

    vocab = process_vocab(data_manager, config)

    gene_ids = vocab.encode(list(data_manager.adata.var_names))
    
    gene_ids = torch.tensor(gene_ids, dtype=torch.long, device=device)
    
    save_path = config.make_path()
    best_loss = float('inf')
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(vf.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.steps, eta_min=config.eta_min)
    
    if config.checkpoint_path != '':
        _, _ = load_checkpoint(config.checkpoint_path, vf, optimizer, scheduler)
    start_iteration = 0 
    vf = accelerator.prepare(vf)
    optimizer, scheduler, dataloader = accelerator.prepare(optimizer,scheduler,dataloader)
    inverse_dict = {v: str(k) for k, v in data_manager.perturbation_dict.items()}
    pbar = tqdm.tqdm(total=config.steps, initial=start_iteration)
    iteration = start_iteration
    while iteration < config.steps:
        for batch_data in dataloader:
            
            source = batch_data['src_cell_data'].squeeze(0)
            target = batch_data['tgt_cell_data'].squeeze(0)
            perturbation_id = batch_data['condition_id'].squeeze(0).to(device)
            if config.perturbation_function == 'crisper':
                perturbation_name = [inverse_dict[int(p_id)] for p_id in perturbation_id[0].cpu().numpy()]
                perturbation_id = torch.tensor(vocab.encode(perturbation_name), dtype=torch.long, device=device)
                perturbation_id = perturbation_id.repeat(source.shape[0],1)
            
            
            set_requires_grad_for_p_only(vf, p_only=config.mode)
            loss = train_step(source, target, perturbation_id, vf, criterion, accelerator, noise_type=config.noise_type, mode=config.mode)
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            
            if iteration % config.print_every == 0:
                save_path_ = os.path.join(save_path, f'iteration_{iteration}')
                os.makedirs(save_path_, exist_ok=True)
                if accelerator.is_main_process:
                    print(f"svaing {iteration}'s checkpoint...")
                    
                    save_checkpoint(
                        model=accelerator.unwrap_model(vf), 
                        optimizer=optimizer, 
                        scheduler=scheduler, 
                        iteration=iteration, 
                        eval_score=None,  # 不需要评估分数
                        save_path=save_path_, 
                        is_best=False
                    )
                eval_score = test(valid_sampler, vf, accelerator, batch_size=config.batch_size, path=save_path_,vocab=vocab)
                
            accelerator.wait_for_everyone()
            
            pbar.update(1)
            pbar.set_description(f'loss: {loss.item():.4f}, iteration: {iteration}')
            iteration += 1
            
            