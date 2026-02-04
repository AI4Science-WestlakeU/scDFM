import scanpy as sc
import pandas as pd
import numpy as np
import json
import torch
import pickle
from typing import Union, Optional
from pathlib import Path
import os
from src.utils._preprocessing import annotate_compounds, get_molecular_fingerprints
from src.data_process._datamanager import DataManager
import jax
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pdb
import tqdm
from random import shuffle
from src.utils.utils import build_gene_coexpression_graph,sorted_pad_mask
# combosciplex url: https://figshare.com/articles/dataset/combosciplex/25062230?file=44229635
# 'norman' url = 'https://dataverse.harvard.edu/api/access/datafile/6154020'

class Data:
    def __init__(self, data_path='../../data'):
        self.data_path = data_path
        if not os.path.exists(data_path):
            raise ValueError(data_path + ' does not exist')
            # os.makedirs(data_path)

        
    def load_data(self, data_name = None, data_path = None):
        self.data_name = data_name
        if data_name in ['norman', 'norman_umi_go_filtered',]:
            self.adata = sc.read_h5ad(os.path.join(self.data_path, data_name + '.h5ad'))
        elif data_name in ['combosciplex', ]:
            self.adata = sc.read_h5ad(os.path.join(self.data_path, data_name + '.h5ad'))
        else:
            raise ValueError(data_name + ' is not a valid data name')
        
    def process_data(self, n_top_genes = 2000,infer_top_gene=1000,split_method='additive',
                     use_negative_edge=True, k=30,
                     **kwargs):
        os.makedirs(os.path.join(self.data_path, self.data_name), exist_ok=True)
        if self.data_name == 'combosciplex':
            
            if os.path.exists(os.path.join(self.data_path, self.data_name, 'processed.h5ad')):
                self.adata = sc.read_h5ad(os.path.join(self.data_path, self.data_name, 'processed.h5ad'))
            else:   

                self.adata.obs["condition"] = self.adata.obs.apply(
                    lambda x: "control" if x["condition"] == "control+control" else x["condition"], axis=1
                )

                self.adata.obs["is_control"] = self.adata.obs.apply(
                    lambda x: True if x["condition"] == "control" else False, axis=1
                )
                
                annotate_compounds(self.adata, compound_keys=["Drug1", "Drug2"])
                get_molecular_fingerprints(self.adata, compound_keys=["Drug1", "Drug2"])
                self.adata.uns["fingerprints"]["control"] = np.zeros(1024)
                
                self.adata.write(os.path.join(self.data_path, self.data_name, 'processed.h5ad'))
            
            self.adata.X = self.adata.layers["counts"].copy()
            sc.pp.normalize_total(self.adata)
            sc.pp.log1p(self.adata)
            sc.pp.highly_variable_genes(self.adata, inplace=True, n_top_genes=n_top_genes)
                
            if 'test_conditions' in kwargs.keys():
                test_conditions = kwargs['test_conditions']
            else:
                test_conditions = ['Panobinostat+Crizotinib', 
                                'Panobinostat+Curcumin', 
                                'Panobinostat+SRT1720', 
                                'Panobinostat+Sorafenib', 
                                'SRT2104+Alvespimycin', 
                                'control+Alvespimycin', 
                                'control+Dacinostat']
                
            self.adata = self.adata[:,self.adata.var['highly_variable']] # filter out low variable genes
            
            self.adata.obs["mode"] = self.adata.obs.apply(lambda x: "test" if x["condition"] in test_conditions else "train", axis=1)
            self.adata_train = self.adata[self.adata.obs["mode"] == "train"]
            self.adata_test = self.adata[(self.adata.obs["mode"] == "test") | (self.adata.obs["condition"]=="control")]
            
            sc.pp.highly_variable_genes(self.adata_test, inplace=True, n_top_genes=infer_top_gene)
            self.adata_test = self.adata_test[:,self.adata_test.var['highly_variable']]
            
            condition = np.unique(list(self.adata.obs['condition']))
            unique_perturbation = []
            np.array([unique_perturbation.extend(perturbation.split('+')) for perturbation in condition])
            unique_perturbation = np.unique(unique_perturbation)
            unique_perturbation.sort()
            self.unique_perturbation = unique_perturbation
            self.perturbation_dict = {perturbation: i for i, perturbation in enumerate(unique_perturbation)}
            # self._val_manager = 
        elif self.data_name == 'norman' or self.data_name == 'norman_umi_go_filtered':
            
            sc.pp.highly_variable_genes(self.adata, inplace=True, n_top_genes=n_top_genes)
            unique_perturbation = []
            [unique_perturbation.extend(perturbation.split('+')) for perturbation in self.adata.obs['condition'].unique()]
            unique_perturbation = np.unique(unique_perturbation)
            
            if self.data_name == 'norman':
                for perturbation in unique_perturbation:
                    if perturbation in self.adata.var_names:
                        self.adata.var.loc[perturbation, 'highly_variable'] = True
                    else:
                        print(f"Warning: {perturbation} is not in the gene names")
                self.adata = self.adata[:,self.adata.var['highly_variable']]
            elif self.data_name == 'norman_umi_go_filtered':
                all_gene_names = list(self.adata.var['gene_name']) + ['ctrl']
                for perturbation in unique_perturbation:
                    if perturbation not in all_gene_names:
                        print(f"Warning: {perturbation} is not in the gene names")
                self.adata.var['highly_variable'] = True

            
            #### for split five times 
            if split_method == 'additive' or 'combinations':
                split_file = os.path.join(self.data_path, self.data_name, 'split_results.pkl')
                if os.path.exists(split_file):
                    with open(split_file, 'rb') as f:
                        self.split_results = pickle.load(f)
                else:
                    perturbations = np.unique(self.adata.obs['condition'])
                    double_perturbation = [p for p in perturbations if 'ctrl' not in p]
                    double_perturbation = np.array(double_perturbation)

                    self.split_results = []
                    
                    for i in range(5):
                        np.random.seed(42 + i)
                        shuffled = double_perturbation.copy()
                        np.random.shuffle(shuffled)
                        
                        split_idx = int(len(shuffled) * 0.3)
                        test_double = shuffled[:split_idx]
                        train_double = shuffled[split_idx:]
                        self.split_results.append({
                            'train': train_double.tolist(),
                            'test': test_double.tolist()
                        })
                    
                    with open(split_file, 'wb') as f:
                        pickle.dump(self.split_results, f)
                    print('split results saved')
                    
            elif split_method == 'unseen':
                split_file = os.path.join(self.data_path, self.data_name, 'split_results_unseen.pkl')
                if os.path.exists(split_file):
                    with open(split_file, 'rb') as f:
                        self.split_results = pickle.load(f)
                else:
                    self.split_results = []
                    for i in range(5):
                        perturbations = np.unique(self.adata.obs['condition'])
                        double_perturbation = [p for p in perturbations if 'ctrl' not in p]
                        single = []
                        [single.extend(p.split('+')) for p in double_perturbation]
                        single = list(set(single))
                    
                        shuffle(single)
                        remove_genes = single[:12]
                        p_count = {}
                        for p in double_perturbation:
                            ps = p.split('+')
                            count = int(ps[0] in remove_genes) + int(ps[1] in remove_genes)
                            p_count[p] = count
                        double_perturbation = [p for p, count in p_count.items() if count > 0]
                        double_perturbation = list(double_perturbation)
                        remove_genes_condition = [p+'+control' for p in remove_genes]
                        double_perturbation.extend(remove_genes_condition)
                        self.split_results.append({
                            'p_count': p_count,
                            'test': double_perturbation
                        })
                    with open(split_file, 'wb') as f:
                        pickle.dump(self.split_results, f)
                    print('split results unseen saved')
            
            if 'fold' in kwargs.keys():
                fold = kwargs['fold']
            else:
                fold = 0
            self.adata.obs['condition'] = self.adata.obs['condition'].str.replace('ctrl', 'control')
            self.adata.obs['Drug1'] = self.adata.obs['condition'].str.split('+').apply(lambda x: x[0])
            self.adata.obs['Drug2'] = self.adata.obs['condition'].str.split('+').apply(lambda x: x[-1])
            self.adata.obs['is_control'] = False
            self.adata.obs.loc[self.adata.obs['control'] == 1, 'is_control'] = True
            self.adata.obs['mode'] = 'train'
            
            
            if split_method == 'combinations':
                self.split_results[fold]['test'] = self.split_results[fold]['test'][:15]
                remove_genes = []
                [remove_genes.extend(p.split('+')) for p in self.split_results[fold]['test']]
                remove_genes = set(remove_genes)
                remove_genes_condition = [p+'+control' for p in remove_genes]
                
                self.split_results[fold]['test'].extend(remove_genes_condition)            
            
            
            self.adata.obs.loc[self.adata.obs['condition'].isin(self.split_results[fold]['test']), 'mode'] = 'test'
            
            self.adata_train = self.adata[self.adata.obs['mode'] == 'train']
            self.adata_test = self.adata[(self.adata.obs['mode'] == 'test') | (self.adata.obs['control'] == 1)]
            
            
            
            sc.pp.highly_variable_genes(self.adata_test, inplace=True, n_top_genes=infer_top_gene)
            self.adata_test = self.adata_test[:,self.adata_test.var['highly_variable']]
            
            condition = np.unique(list(self.adata.obs['condition']))
            unique_perturbation = []
            np.array([unique_perturbation.extend(perturbation.split('+')) for perturbation in condition])
            unique_perturbation = np.unique(unique_perturbation)
            unique_perturbation.sort()
            self.unique_perturbation = unique_perturbation
            self.perturbation_dict = {perturbation: i for i, perturbation in enumerate(unique_perturbation)}
            
        else:
            raise ValueError(self.data_name + ' is not a valid data name')
        
        if 'fold' in kwargs.keys():
            fold = kwargs['fold']
        else:
            fold = 0
        if use_negative_edge:
            mask_path = os.path.join(self.data_path, self.data_name,'mask_fold_'+str(fold)+'topk_'+str(k)+split_method+'_negative_edge'+'.pt')
        else:
            mask_path = os.path.join(self.data_path, self.data_name,'mask_fold_'+str(fold)+'topk_'+str(k)+split_method+'.pt')
        if os.path.exists(mask_path):
            self.mask = torch.load(mask_path)
        else:
            X = self.adata_train.X.toarray()
            mask = build_gene_coexpression_graph(X,
                method="pearson",
                wgcna_beta=None,
                sparsify="topk",
                k=k,
                use_negative_edge=use_negative_edge)
            mask = sorted_pad_mask(mask, pad_size=4, gene_names=list(self.adata_train.var_names))
            torch.save(mask, mask_path)
            print('mask saved')
        
        
    def load_flow_data(self, batch_size = 128):
        if self.data_name == 'combosciplex':
            train_sampler = TrainSampler(self.data_name, self.adata_train, ["Drug1", "Drug2"], self.perturbation_dict)
            test_sampler = TestDataset(self.data_name, self.adata_test, ["Drug1", "Drug2"], self.perturbation_dict)
            
            return train_sampler , test_sampler, []
        elif self.data_name == 'norman' or self.data_name == 'norman_umi_go_filtered':
            train_sampler = TrainSampler(self.data_name, self.adata_train, ["Drug1", "Drug2"], self.perturbation_dict)
            test_sampler = TestDataset(self.data_name, self.adata_test, ["Drug1", "Drug2"], self.perturbation_dict)
            return train_sampler , test_sampler, []
        else:
            raise ValueError(self.data_name + ' is not a valid data name')
            

    def pretrain_data(self, batch_size = 128):
        if self.data_name == 'combosciplex':
            
            self.pretrain_train_data = PretrainData(self.adata_train, self.perturbation_dict)
            self.pretrain_train_data_loader = DataLoader(self.pretrain_train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
            self.pretrain_test_data = PretrainData(self.adata_test, self.perturbation_dict)
            self.pretrain_test_data_loader = DataLoader(self.pretrain_test_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
            return self.pretrain_train_data_loader, self.pretrain_test_data_loader
        else:
            raise ValueError(self.data_name + ' is not a valid data name')
        
class TrainSampler:
    def __init__(self, data_name, adata: sc.AnnData, perturbation_covariates: list[str], perturbation_dict: dict,):
        self.data_name = data_name
        self.adata = adata
        self.perturbation_covariates = perturbation_covariates
        self.adata.obs['perturbation_covariates'] = self.adata.obs[perturbation_covariates].apply(lambda x: '+'.join(x), axis=1)
        self._perturbation_covariates = adata.obs['perturbation_covariates'].unique()
        
        self._perturbation_covariates = self._perturbation_covariates[self._perturbation_covariates != 'control+control']
        
        self._perturbation_covariates.sort()
        self.perturbation_covariates_dict = {perturbation: i for i, perturbation in enumerate(self._perturbation_covariates)}
        
        perturbation_covariates_id = [adata.obs[perturbation_covariates[i]].apply(lambda x: perturbation_dict[x])
                                    for i in range(len(perturbation_covariates))]
        self.perturbation_covariates_id = np.array(perturbation_covariates_id).T
        
        
        self.cells_name = self.adata.obs_names
        
        
    def get_batch(self, batch_size: int, same_perturbation: bool = True):
        if same_perturbation:
            # random sample a perturbation from self._perturbation_covariates.
            # the last one is control
            perturbation_idx = np.random.choice(len(self._perturbation_covariates), 1)[0]
            
            perturbation_id = self._perturbation_covariates[perturbation_idx]
            
            # get the target data
            tgt_idx = (self.adata.obs['perturbation_covariates'] == perturbation_id).to_numpy().nonzero()[0]
            tgt_batch_idx = np.random.choice(tgt_idx, batch_size)
            
            tgt_batch = torch.from_numpy(self.adata.X[tgt_batch_idx].toarray())
            
            # get data from control
            src_idx = (self.adata.obs['perturbation_covariates'] == 'control+control').to_numpy().nonzero()[0]
            src_batch_idx = np.random.choice(src_idx, batch_size)
            
            src_batch = torch.from_numpy(self.adata.X[src_batch_idx].toarray())
            
            return {
                'src_cell_data': src_batch,
                'tgt_cell_data': tgt_batch,
                'src_cell_id': self.cells_name[src_batch_idx],
                'tgt_cell_id': self.cells_name[tgt_batch_idx],
                'condition_id': self.perturbation_covariates_id[tgt_batch_idx],
            }
            
        else:
            raise ValueError('same_perturbation must be True')
            
class TestDataset:
    def __init__(self, data_name,adata: sc.AnnData, perturbation_covariates: list[str], perturbation_dict: dict,):
        self.data_name = data_name
        self.adata = adata
        self.perturbation_covariates = perturbation_covariates
        self.adata.obs['perturbation_covariates'] = self.adata.obs[perturbation_covariates].apply(lambda x: '+'.join(x), axis=1)
        self._perturbation_covariates = adata.obs['perturbation_covariates'].unique()
        
        self._perturbation_covariates = self._perturbation_covariates[self._perturbation_covariates != 'control+control']
        
        self._perturbation_covariates.sort()
        self.perturbation_covariates_dict = {perturbation: i for i, perturbation in enumerate(self._perturbation_covariates)}
        
        perturbation_covariates_id = [adata.obs[perturbation_covariates[i]].apply(lambda x: perturbation_dict[x])
                                    for i in range(len(perturbation_covariates))]
        self.perturbation_covariates_id = np.array(perturbation_covariates_id).T
        
        
        self.cells_name = self.adata.obs_names
        
    def get_control_data(self,):
        control_data = self.adata[self.adata.obs['is_control']]
        return {
            'src_cell_data': torch.from_numpy(control_data.X.toarray()),
            'src_cell_id': control_data.obs_names,
            'condition_id': torch.tensor(self.perturbation_covariates_id[self.adata.obs['is_control']]),
        }
    
    def get_perturbation_data(self, perturbation: str):
        perturbation_data = self.adata[self.adata.obs['perturbation_covariates'] == perturbation]
        return {
            'tgt_cell_data': torch.from_numpy(perturbation_data.X.toarray()),
            'tgt_cell_id': perturbation_data.obs_names,
            'condition_id': torch.tensor(self.perturbation_covariates_id[self.adata.obs['perturbation_covariates'] == perturbation]),
        }
        
    
    
class PerturbationDataset(Dataset):
    def __init__(self, sampler: TrainSampler, batch_size: int):
        self.sampler = sampler
        self.batch_size = batch_size
        self.perturbations = sampler._perturbation_covariates
        
        self.control_idx = (sampler.adata.obs['perturbation_covariates'] == 'control+control').to_numpy().nonzero()[0]
        
    def __len__(self):
        
        return len(self.perturbations) * 1000  
    
    def __getitem__(self, idx):
        # 随机选一个 perturbation
        perturbation_idx = np.random.choice(len(self.perturbations), 1)[0]
        perturbation_id = self.perturbations[perturbation_idx]

        # target batch
        tgt_idx = (self.sampler.adata.obs['perturbation_covariates'] == perturbation_id).to_numpy().nonzero()[0]
        tgt_batch_idx = np.random.choice(tgt_idx, self.batch_size)
        
        # source (control) batch
        src_batch_idx = np.random.choice(self.control_idx, self.batch_size)
        if hasattr(self.sampler.adata.X[src_batch_idx], "toarray"):
            src_batch = torch.from_numpy(self.sampler.adata.X[src_batch_idx].toarray())
            tgt_batch = torch.from_numpy(self.sampler.adata.X[tgt_batch_idx].toarray())
        else:
            src_batch = torch.from_numpy(self.sampler.adata.X[src_batch_idx])
            tgt_batch = torch.from_numpy(self.sampler.adata.X[tgt_batch_idx])
        
        return {
            'src_cell_data': src_batch,
            'tgt_cell_data': tgt_batch,
            'src_cell_id': list(self.sampler.cells_name[src_batch_idx]),
            'tgt_cell_id': list(self.sampler.cells_name[tgt_batch_idx]),
            'condition_id': torch.tensor(self.sampler.perturbation_covariates_id[tgt_batch_idx]),
        }
class BinDiscretizer:
    """
    data = np.random.exponential(scale=2.0, size=1000)

    bd = BinDiscretizer(n_bins=200)
    bd.fit(data)

    bd.save_edges('./data/combosciplex/bin_discretizer_edges.pkl')

    new_bd = BinDiscretizer(n_bins=200)
    new_bd.load_edges('./data/combosciplex/bin_discretizer_edges.pkl')

    binned = bd.transform(data)

    recon = bd.inverse_transform(binned, random=False)
    
    Note: 0 is treated as a separate class (class 0), and non-zero values are discretized into classes 1 to n_bins.
    """
    def __init__(self, n_bins: int, strategy: str = "quantile"):
        self.n_bins = n_bins
        self.strategy = strategy
        self.edges = None  # will be (n_bins + 1, ) array

    def fit(self, data: Union[np.ndarray, torch.Tensor]):
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        data = data.flatten()
        data = data[data > 0]  # exclude zeros from fitting

        if len(data) == 0:
            raise ValueError("No non-zero entries in data to fit.")

        if self.strategy == "quantile":
            self.edges = np.quantile(data, np.linspace(0, 1, self.n_bins + 1))
        elif self.strategy == "uniform":
            self.edges = np.linspace(data.min(), data.max(), self.n_bins + 1)
        else:
            raise ValueError(f"Unknown strategy {self.strategy}")

    def transform(self, data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if self.edges is None:
            raise RuntimeError("Call fit() before transform().")

        is_torch = isinstance(data, torch.Tensor)
        if is_torch:
            orig_dtype = data.dtype
            data = data.detach().cpu().numpy()

        out = np.zeros_like(data, dtype=np.int64)
        mask = data > 0

        # Digitize non-zero entries: 0 remains 0, non-zero values get classes 1 to n_bins
        if np.any(mask):
            # np.digitize returns 0-based indices for the bins
            # We want to map these to 1-based class indices
            digitized = np.digitize(data[mask], self.edges[1:-1])
            # Convert 0-based bin indices to 1-based class indices
            # digitized=0 means it's in the first bin, which should be class 1
            # digitized=1 means it's in the second bin, which should be class 2, etc.
            out[mask] = digitized + 1

        if is_torch:
            return torch.from_numpy(out).to(dtype=torch.int64)
        return out

    def inverse_transform(self, digitized: Union[np.ndarray, torch.Tensor], random: bool = False) -> Union[np.ndarray, torch.Tensor]:
        if self.edges is None:
            raise RuntimeError("Call fit() before inverse_transform().")

        is_torch = isinstance(digitized, torch.Tensor)
        if is_torch:
            orig_dtype = digitized.dtype
            digitized = digitized.detach().cpu().numpy()

        out = np.zeros_like(digitized, dtype=np.float64)
        mask = digitized > 0

        if np.any(mask):
            ids = digitized[mask]
            # Ensure ids are within valid range (1 to n_bins)
            ids = np.clip(ids, 1, self.n_bins)
            # Convert 1-based class indices back to 0-based bin indices
            bin_ids = ids - 1
            lefts = self.edges[bin_ids]
            rights = self.edges[bin_ids + 1]

            if random:
                out[mask] = np.random.uniform(lefts, rights)
            else:
                out[mask] = (lefts + rights) / 2

        if is_torch:
            return torch.from_numpy(out).to(dtype=torch.float64)
        return out
    
    def save_edges(self, filepath: Union[str, Path]):
        """Save edges to a file"""
        if self.edges is None:
            raise RuntimeError("No edges to save. Call fit() first.")
        
        filepath = Path(filepath)
        with open(filepath, 'wb') as f:
            pickle.dump({'edges': self.edges, 'n_bins': self.n_bins}, f)
            
    def load_edges(self, filepath: Union[str, Path]):
        """Load edges from a file"""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File {filepath} not found")
            
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            loaded_n_bins = data['n_bins']
            if loaded_n_bins != self.n_bins:
                raise ValueError(f"Loaded n_bins ({loaded_n_bins}) does not match initialized n_bins ({self.n_bins})")
            self.edges = data['edges']
    
class PretrainData(Dataset):
    def __init__(self, adata: sc.AnnData, drug_dict: dict):
        self.adata = adata
        self.drug_dict = drug_dict
        self.X = torch.from_numpy(adata.X.toarray())
        self.cell_id = adata.obs_names
        # self.drug1 = torch.tensor(np.array(adata.obs['Drug1'].apply(lambda x: drug_dict[x])))
        # self.drug2 = torch.tensor(np.array(adata.obs['Drug2'].apply(lambda x: drug_dict[x])))
        
    def __len__(self):
        return len(self.adata)
    
    def __getitem__(self, idx):
        return {
            'values' : self.X[idx], 
            'cell_id': self.cell_id[idx],
        }
    
            
class FlowMatchingDataset(Dataset):
    """PyTorch Dataset for flow matching training data"""
    
    def __init__(self, jax_sampler, num_samples=10000, seed=42):
        """
        Args:
            jax_sampler: JAX-based TrainSampler
            num_samples: Number of samples to generate per epoch
            seed: Random seed
        """
        self.jax_sampler = jax_sampler
        self.num_samples = num_samples
        self.rng = jax.random.PRNGKey(seed)
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        """Sample a batch from the JAX sampler"""
        # Generate new random key for each sample
        self.rng, sample_key = jax.random.split(self.rng)
        
        # Sample from JAX sampler
        sample = self.jax_sampler.sample(sample_key)
        
        # Convert JAX arrays to PyTorch tensors
        src_cell_data = torch.from_numpy(np.array(sample['src_cell_data'])).float()
        tgt_cell_data = torch.from_numpy(np.array(sample['tgt_cell_data'])).float()
        
        sample['src_cell_id']
        sample['tgt_cell_id']
        # Convert condition embedding if available
        condition_data = None
        if 'condition' in sample:
            condition_data = {
                key: torch.from_numpy(np.array(val)).float()
                for key, val in sample['condition'].items()
            }
        
        # Convert condition_id embedding if available
        condition_id = None
        if 'condition_id' in sample:
            condition_id = torch.from_numpy(np.array(sample['condition_id'])).long()
        
        return {
            'src_cell_data': src_cell_data,
            'tgt_cell_data': tgt_cell_data,
            'condition': condition_data,
            'condition_id': condition_id,
        }
        
if __name__ == "__main__":
    data = Data(data_path='./data')
    data.load_data(data_name='combosciplex')
    data.process_data()
    
    
    