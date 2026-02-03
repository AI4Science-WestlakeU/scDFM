
import sys, os
import scipy.io
import h5py
import pdb
import pickle

import torch
import numpy as np
import torch.nn as nn
from torch_geometric.data import Dataset

from typing import Tuple
from einops import rearrange, repeat
from IPython import embed


class Advection(Dataset):
    '''
    x: u in Burgers' equation
    y: u_1, u_2, ..., u_t, (Nt, Nx, 1)
    '''
    def __init__(
        self,
        # basic arguments
        dataset_name="Advection",
        dataset_path=None,
        mode = 'train',# 'test'

        # specific arguments
        input_steps=1,
        output_steps=1,
        time_interval=1,
        simutime_steps=80,
        rescaler=4,
    ):
        # arguments
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.mode = mode

        self.input_steps = input_steps
        self.output_steps = output_steps
        self.time_interval = time_interval
        self.simutime_steps = simutime_steps
        self.rescaler = rescaler


        #load raw data
        if self.mode=='train':
            path=dataset_path+'/train'
        elif self.mode=='test':
            path=dataset_path+'/test'
        self.dataset_cache = torch.load(path)
        print("Load dataset {}".format(path))
        
        # basic preprocessing
        self.dataset_cache = self.dataset_cache[:,:81,::8]  # downsample
        self.dataset_cache = self.dataset_cache/self.rescaler # normalize

        # basic features of dataset
        self.n_simu = len(self.dataset_cache) # [9000,201,1024->128]
        self.nx = self.dataset_cache.shape[-1] # 128
        self.t_cushion_input = self.input_steps * self.time_interval if self.input_steps * self.time_interval > 1 else 1
        self.t_cushion_output = self.output_steps * self.time_interval if self.output_steps * self.time_interval > 1 else 1
        self.time_stamps_effective = (self.simutime_steps+1 - self.t_cushion_input - self.t_cushion_output + self.time_interval) // self.time_interval

        super(Advection, self).__init__()

    def len(self): # must be implemented
        return int((self.time_stamps_effective * self.n_simu)*1.0)

    def get(self, idx, use_normalized=True): # must be implemented
        '''
        data:
            input: [1,T,s] repeat u0 for T times
            target: [1,T,s],trajectory of u
        '''
        # get id
        sim_id, time_id = divmod(idx, self.time_stamps_effective)
        if time_id+self.output_steps>self.simutime_steps:
            time_id =0
        data_traj=self.dataset_cache[sim_id]
        u = torch.tensor(data_traj, dtype=torch.float32)

        # get input and target
        u0=u[0]
        input = torch.zeros((1,self.simutime_steps, self.nx)) # [c=1,T,s]
        u0_repeat = repeat(u0, 's -> c s', c=self.simutime_steps)
        input[0] = u0_repeat
        target = u[1:].reshape(-1,self.simutime_steps, self.nx) # [c=1,T,s]

        data = (
            input, #[c,T,s]
            target # [c,T,s]
        )

        return data

if __name__ == '__main__':
    # Import EXP_PATH for proper path handling
    from standard_repo_module.filepath import EXP_PATH
    import os
    
    test_dataset = Advection(
        dataset_name="Advection",
        dataset_path=os.path.join(EXP_PATH, 'standard_repo/dataset/advection'),
        mode='test',
        input_steps=1,
        output_steps=80,
        time_interval=1,
        simutime_steps=80,
        rescaler=4,
    )
    print(len(test_dataset))