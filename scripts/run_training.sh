#!/bin/bash
# Training script for standard_repo project using tyro configuration
# This script activates the conda environment and runs the training demo

# Activate the conda environment
source /opt/conda/bin/activate
conda activate ideEnv

# Run the python training script with tyro configuration
# You can now pass arguments like:
# python standard_repo_module/train/train_demo.py --training.date-exp 2024-09-08 --training.epochs 50
python standard_repo_module/train/train_demo.py
