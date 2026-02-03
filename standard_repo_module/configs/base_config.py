"""
Configuration classes for standard_repo using tyro.

This module defines dataclass-based configuration structures that provide
type safety, IDE support, and automatic CLI generation through tyro.
"""
from dataclasses import dataclass
from typing import Optional, Union
import os
from pathlib import Path


@dataclass
class BaseConfig:
    """Base configuration class with common parameters.
    
    This class provides common configuration parameters shared across
    training and evaluation configurations.
    """
    
    # Experiment identification
    date_exp: str = "2024-09-08"  # Experiment date in YYYY-MM-DD format
    exp_name: str = "demo_experiment"  # Experiment name for identification
    
    # Paths
    dataset_path: str = "standard_repo/dataset/advection"  # Relative path to dataset
    results_path: str = "results"  # Base results directory
    config_path: Optional[str] = None  # Path to config file for loading parameters
    
    # Hardware and reproducibility
    gpu_id: int = 0  # GPU device ID to use
    seed: int = 42  # Random seed for reproducibility
    num_workers: int = 0  # Number of dataloader workers
    
    def __post_init__(self) -> None:
        """Post-initialization to set up derived paths and directories.
        
        Sets up experiment directory structure with hash-based naming
        and ensures all necessary directories exist.
        """
        # Import here to avoid circular imports
        from standard_repo_module.filepath import EXP_PATH
        from standard_repo_module.utils.utils import get_config_hash
        
        # Generate config hash for experiment naming
        config_hash = get_config_hash(self)[:8]
        
        # Set up full paths using EXP_PATH
        self.results_path = os.path.join(EXP_PATH, "results", self.date_exp, f"{self.exp_name}_{config_hash}")
        self.dataset_path = os.path.join(EXP_PATH, self.dataset_path)
        
        if self.config_path:
            self.config_path = os.path.join(EXP_PATH, self.config_path)
        
        # Create experiment directory structure
        subdirs = ['checkpoints', 'logs', 'plots', 'inference_results']
        for subdir in subdirs:
            os.makedirs(os.path.join(self.results_path, subdir), exist_ok=True)


@dataclass
class TrainingConfig(BaseConfig):
    """Training configuration with all training-specific parameters.
    
    Contains hyperparameters and settings specific to model training,
    including learning rates, batch sizes, and checkpoint management.
    """
    
    # Training hyperparameters
    epochs: int = 100  # Number of training epochs
    save_every: int = 20  # Save model checkpoint every N epochs
    train_batch_size: int = 256  # Training batch size
    test_batch_size: int = 256  # Testing/validation batch size
    lr: float = 0.001  # Learning rate for optimizer
    
    # Model and checkpoint management
    checkpoint_path: Optional[str] = None  # Path to load pretrained checkpoint
    
    # Logging and monitoring
    is_use_tfb: bool = True  # Whether to use TensorBoard logging
    
    def __post_init__(self) -> None:
        """Post-initialization for training configuration.
        
        Sets up training-specific paths and ensures checkpoint path
        is properly resolved if provided.
        """
        super().__post_init__()
        
        # Handle checkpoint path resolution
        if self.checkpoint_path:
            from standard_repo_module.filepath import EXP_PATH
            self.checkpoint_path = os.path.join(EXP_PATH, self.checkpoint_path)


@dataclass
class EvaluationConfig(BaseConfig):
    """Evaluation configuration with evaluation-specific parameters."""
    
    # Evaluation parameters
    eval_batch_size: int = 256
    checkpoint_path: str = "standard_repo/results/2024-08-05/training_demo/model_epoch_20.pth"
    
    def __post_init__(self):
        """Post-initialization for evaluation config."""
        super().__post_init__()
        
        # Handle checkpoint path
        from standard_repo_module.filepath import EXP_PATH
        self.checkpoint_path = os.path.join(EXP_PATH, self.checkpoint_path)


@dataclass
class DataConfig:
    """Data-specific configuration parameters."""
    
    dataset_name: str = "Advection"
    mode: str = "train"  # 'train', 'test', 'val'
    
    # Data processing parameters
    input_steps: int = 1
    output_steps: int = 80
    time_interval: int = 1
    simutime_steps: int = 80
    rescaler: int = 4


@dataclass
class ModelConfig:
    """Model-specific configuration parameters."""
    
    # Add model-specific parameters here
    # This can be extended based on your model requirements
    model_type: str = "Net_demo"
    
    # Model architecture parameters can be added here
    # hidden_dim: int = 128
    # num_layers: int = 3
    # etc.


@dataclass
class FullTrainingConfig:
    """Complete training configuration combining all sub-configs."""
    
    training: TrainingConfig
    data: DataConfig
    model: ModelConfig
    
    def __init__(
        self,
        # Training parameters
        date_exp: str = "2024-09-08",
        exp_name: str = "training_demo_test",
        epochs: int = 10,
        save_every: int = 1,
        train_batch_size: int = 512,
        test_batch_size: int = 512,
        lr: float = 0.001,
        checkpoint_path: Optional[str] = None,
        gpu_id: int = 0,
        seed: int = 42,
        num_workers: int = 0,
        is_use_tfb: bool = True,
        
        # Data parameters
        dataset_name: str = "Advection",
        mode: str = "train",
        input_steps: int = 1,
        output_steps: int = 80,
        time_interval: int = 1,
        simutime_steps: int = 80,
        rescaler: int = 4,
        
        # Model parameters
        model_type: str = "Net_demo",
    ):
        self.training = TrainingConfig(
            date_exp=date_exp,
            exp_name=exp_name,
            epochs=epochs,
            save_every=save_every,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            lr=lr,
            checkpoint_path=checkpoint_path,
            gpu_id=gpu_id,
            seed=seed,
            num_workers=num_workers,
            is_use_tfb=is_use_tfb,
        )
        
        self.data = DataConfig(
            dataset_name=dataset_name,
            mode=mode,
            input_steps=input_steps,
            output_steps=output_steps,
            time_interval=time_interval,
            simutime_steps=simutime_steps,
            rescaler=rescaler,
        )
        
        self.model = ModelConfig(
            model_type=model_type,
        )


@dataclass
class FullEvaluationConfig:
    """Complete evaluation configuration combining all sub-configs."""
    
    evaluation: EvaluationConfig
    data: DataConfig
    model: ModelConfig
    
    def __init__(
        self,
        # Evaluation parameters
        date_exp: str = "2024-08-05",
        exp_name: str = "eval_demo_test",
        eval_batch_size: int = 512,
        checkpoint_path: str = "standard_repo/results/2024-08-05/training_demo/model_epoch_20.pth",
        gpu_id: int = 0,
        seed: int = 42,
        
        # Data parameters
        dataset_name: str = "Advection",
        mode: str = "test",
        input_steps: int = 1,
        output_steps: int = 80,
        time_interval: int = 1,
        simutime_steps: int = 80,
        rescaler: int = 4,
        
        # Model parameters
        model_type: str = "Net_demo",
    ):
        self.evaluation = EvaluationConfig(
            date_exp=date_exp,
            exp_name=exp_name,
            eval_batch_size=eval_batch_size,
            checkpoint_path=checkpoint_path,
            gpu_id=gpu_id,
            seed=seed,
        )
        
        self.data = DataConfig(
            dataset_name=dataset_name,
            mode=mode,
            input_steps=input_steps,
            output_steps=output_steps,
            time_interval=time_interval,
            simutime_steps=simutime_steps,
            rescaler=rescaler,
        )
        
        self.model = ModelConfig(
            model_type=model_type,
        )
