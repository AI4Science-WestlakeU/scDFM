#!/usr/bin/env python
# coding: utf-8

# # Import package

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
from torch.utils.tensorboard import SummaryWriter
import datasets 
import numpy as np
import tyro
from typing import Tuple

# Custom packages
from standard_repo.data.data_demo import Advection
from standard_repo.model.model_demo import Net_demo
from standard_repo.utils.utils import set_seed, draw_loss, save_config_from_tyro, setup_experiment_directory
from standard_repo_module.configs.base_config import FullTrainingConfig
# Path
from standard_repo.filepath import EXP_PATH, SRC_PATH, PARENT_WP


# # Configuration using tyro

# In[ ]:

def setup_training_config() -> Tuple[FullTrainingConfig, bool]:
    """Set up training configuration with proper experiment directory.
    
    Returns:
        Tuple of (configuration_object, is_jupyter_environment)
        
    Raises:
        OSError: If experiment directory creation fails
    """
    # Check if running in Jupyter environment
    try:
        get_ipython().run_line_magic('matplotlib', 'inline')
        # For Jupyter notebook, use default configuration
        config = FullTrainingConfig()
        is_jupyter = True
    except NameError:
        # For command line, use tyro to parse arguments
        config = tyro.cli(FullTrainingConfig)
        is_jupyter = False
    
    # Set up experiment directory with hash-based naming
    exp_dir, config_hash = setup_experiment_directory(config, EXP_PATH + "/results")
    config.training.results_path = exp_dir
    
    # Save configuration to experiment directory
    save_config_from_tyro(config, exp_dir)
    
    return config, is_jupyter

# Get configuration and set up experiment
config, is_jupyter = setup_training_config()
# set up logging
current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_filename = os.path.join(config.training.results_path, "training_{}.log".format(current_time))
if config.training.is_use_tfb:
    writer = SummaryWriter(log_dir=config.training.results_path, filename_suffix="training_tf{}".format(current_time))
logging.basicConfig(filename=log_filename, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
logging.info('Configuration: {}'.format(config))
# set device and seed
device = torch.device("cuda:"+str(config.training.gpu_id) if torch.cuda.is_available() else "cpu")
set_seed(config.training.seed)



# # Basic func

# In[ ]:


def cycle(dl):
    while True:
        for data in dl:
            yield data
def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


# # Load data and initialize model, model optimizer, loss function

# In[ ]:


# load dataset
train_dataset = Advection(
        dataset_name=config.data.dataset_name,
        dataset_path=config.training.dataset_path,
        mode='train',
        input_steps=config.data.input_steps,
        output_steps=config.data.output_steps,
        time_interval=config.data.time_interval,
        simutime_steps=config.data.simutime_steps,
        rescaler=config.data.rescaler,
    )
test_dataset = Advection(
        dataset_name=config.data.dataset_name,
        dataset_path=config.training.dataset_path,
        mode='test',
        input_steps=config.data.input_steps,
        output_steps=config.data.output_steps,
        time_interval=config.data.time_interval,
        simutime_steps=config.data.simutime_steps,
        rescaler=config.data.rescaler,
    )
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.training.train_batch_size, shuffle=True, pin_memory=True, num_workers=config.training.num_workers)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.training.test_batch_size, shuffle=False, pin_memory=True, num_workers=config.training.num_workers)
logging.info(f"Data loaded from {config.training.dataset_path}")

# configure model
model = Net_demo().to(device)
if config.training.checkpoint_path is not None:
    model.load_state_dict(torch.load(config.training.checkpoint_path))
    logging.info(f"Checkpoint {config.training.checkpoint_path} loaded")

# configure optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
criterion = nn.MSELoss()


# # Training model

# In[ ]:


logging.info("Start training on ",torch.cuda.get_device_name())
num_params = sum(p.numel() for p in model.parameters())
logging.info("Number of parameters: {}".format(num_params))
logging.info("number of batch in train_loader: ", len(train_dataloader))
print("number of batch in test_loader: ", len(test_dataloader))
start_time = time.time()
training_loss_list = []
test_loss_list = []
best_epoch = 0

# training loop
for epoch in tqdm.tqdm(range(1, config.training.epochs + 1)):
    model.train()
    total_loss = 0.
    best_test_loss = 1e9
    for batch_idx, data in enumerate(train_dataloader):
        input, target = data
        input = input.to(device)
        target = target.to(device)

        optimizer.zero_grad()
        output = model(input)  # [B,1,T,s] => [B,1,T,s]
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()
        total_loss += loss.sum().item()
        torch.cuda.empty_cache()
    scheduler.step()
    
    average_loss = total_loss / len(train_dataloader)
    training_loss_list.append(average_loss)
    logging.info("Training epoch {}, average loss: {}".format(epoch, average_loss))
    
    with torch.no_grad():
        for test_data in test_dataloader:
            break
        input, target = test_data
        input = input.to(device)
        target = target.to(device)
        output = model(input)  # [B,1,T,s] => [B,1,T,s]
        loss = criterion(output, target)
        test_loss = loss.sum().item()
        test_loss_list.append(test_loss)
        if test_loss < best_test_loss:
            best_epoch = epoch
            best_test_loss = test_loss
        logging.info("Testing epoch {}, loss: {}".format(epoch, test_loss))
    
    if config.training.is_use_tfb:
        writer.add_scalar("Loss/train", average_loss, epoch)
        writer.add_scalar("Loss/test", test_loss, epoch)
    
    if epoch % config.training.save_every == 0:
        checkpoint_path = os.path.join(config.training.results_path, "checkpoints", f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        plot_path = os.path.join(config.training.results_path, "plots", "loss_curves.png")
        draw_loss(training_loss_list, test_loss_list, plot_path)

end_time = time.time()
logging.info(f"Training complete, best epoch is {best_epoch}, time cost is {end_time-start_time}s")
logging.info("Results saved at {}".format(config.training.results_path))
if config.training.is_use_tfb:
    writer.close()

# # Evaluate the model
