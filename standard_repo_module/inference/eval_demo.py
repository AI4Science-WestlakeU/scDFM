#!/usr/bin/env python
# coding: utf-8

# # Load packages
# 

# In[ ]:


try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except:
    pass
# general package

import matplotlib.pyplot as plt
import os
import sys
import tqdm
from einops import rearrange, repeat
import logging
import datetime  
import time
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets 
import numpy as np
import tyro

# Custom packages, model, data, utils
from standard_repo.data.data_demo import Advection
from standard_repo.model.model_demo import Net_demo
from standard_repo.utils.utils import set_seed, caculate_confidence_interval, save_config_from_tyro
from standard_repo_module.configs.base_config import FullEvaluationConfig
# Path
from standard_repo.filepath import EXP_PATH, SRC_PATH, PARENT_WP


# # Configuration using tyro

# In[ ]:

def main():
    """Main evaluation function with tyro configuration."""
    
    # Check if running in Jupyter
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        # For Jupyter notebook, use default configuration
        config = FullEvaluationConfig()
        is_jupyter = True
    except:
        # For command line, use tyro to parse arguments
        config = tyro.cli(FullEvaluationConfig)
        is_jupyter = False
    
    # Save configuration
    save_config_from_tyro(config, config.evaluation.results_path)
    
    return config, is_jupyter

# Get configuration
config, is_jupyter = main()

# Set up logging
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_filename = os.path.join(config.evaluation.results_path, "evaluation_{}.log".format(current_time))
logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('Configuration: {}'.format(config))

# Set device and seed
device = torch.device("cuda:"+str(config.evaluation.gpu_id) if torch.cuda.is_available() else "cpu")
set_seed(config.evaluation.seed)


# # Load data and Cingure model

# In[ ]:


# load dataset
eval_dataset = Advection(
        dataset_name=config.data.dataset_name,
        dataset_path=config.evaluation.dataset_path,
        mode=config.data.mode,
        input_steps=config.data.input_steps,
        output_steps=config.data.output_steps,
        time_interval=config.data.time_interval,
        simutime_steps=config.data.simutime_steps,
        rescaler=config.data.rescaler,
    )
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=config.evaluation.eval_batch_size, shuffle=False, pin_memory=True, num_workers=8)
logging.info(f"Data loaded from {config.evaluation.dataset_path}")

# configure model
model = Net_demo().to(device)
if config.evaluation.checkpoint_path is not None:
    model.load_state_dict(torch.load(config.evaluation.checkpoint_path))
    logging.info(f"Checkpoint {config.evaluation.checkpoint_path} loaded")

criterion = nn.MSELoss()


# # Evaluation

# In[ ]:


logging.info("Start evaluation on {}".format(torch.cuda.get_device_name()))
num_params = sum(p.numel() for p in model.parameters())
logging.info("Number of parameters: {}".format(num_params))
start_time = time.time()
bs = config.evaluation.eval_batch_size

with torch.no_grad():
    for eval_data in eval_dataloader:
        break
    input, target = eval_data
    input = input.to(device)
    target = target.to(device)
    output = model(input)  # [B,1,T,s] => [B,1,T,s]

end_time = time.time()
relative_mse = torch.norm(target - output) / torch.norm(target)
target = target.reshape(bs, -1)
output = output.reshape(bs, -1)
mse_batch = ((target - output) ** 2).mean(-1)  # [B,]
mean, std_dev, margin_of_error, min_value = caculate_confidence_interval(mse_batch)

logging.info("Evaluated checkpoint: {}".format(config.evaluation.checkpoint_path))
logging.info("Relative MSE: {:.8f}".format(relative_mse))
logging.info("MSE: {:.8f}".format(mean))
logging.info("MSE std: {:.8f}".format(std_dev))
logging.info("MSE margin of error: {:.8f}".format(margin_of_error))
logging.info(f"Evaluation complete, time cost is {end_time-start_time}s")
logging.info("Results saved at {}".format(config.evaluation.results_path))

