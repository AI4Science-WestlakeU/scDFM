import torch
import numpy as np
import yaml
import argparse
from datetime import datetime
import os
import sys
from termcolor import colored
import pdb
import matplotlib.pyplot as plt
import hashlib
import json
from typing import Dict, Any, Tuple, Optional, Union
from dataclasses import asdict
COLOR_LIST = ["b", "r", "g", "y", "c", "m", "skyblue", "indigo", "goldenrod", "salmon", "pink",
                  "silver", "darkgreen", "lightcoral", "navy", "orchid", "steelblue", "saddlebrown", 
                  "orange", "olive", "tan", "firebrick", "maroon", "darkslategray", "crimson", "dodgerblue", "aquamarine",
             "b", "r", "g", "y", "c", "m", "skyblue", "indigo", "goldenrod", "salmon", "pink",
                  "silver", "darkgreen", "lightcoral", "navy", "orchid", "steelblue", "saddlebrown", 
                  "orange", "olive", "tan", "firebrick", "maroon", "darkslategray", "crimson", "dodgerblue", "aquamarine"]

#basic utils
class Printer(object):
    ## Example, to print code running time between two p.print() calls
    # p.print(f"test_start", tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=1, precision="millisecond", is_silent=False)
    def __init__(self, is_datetime=True, store_length=100, n_digits=3):
        """
        Args:
            is_datetime: if True, will print the local date time, e.g. [2021-12-30 13:07:08], as prefix.
            store_length: number of past time to store, for computing average time.
        Returns:
            None
        """
        
        self.is_datetime = is_datetime
        self.store_length = store_length
        self.n_digits = n_digits
        self.limit_list = []

    def print(self, item, tabs=0, is_datetime=None, banner_size=0, end=None, avg_window=-1, precision="second", is_silent=False):
        if is_silent:
            return
        string = ""
        if is_datetime is None:
            is_datetime = self.is_datetime
        if is_datetime:
            str_time, time_second = get_time(return_numerical_time=True, precision=precision)
            string += str_time
            self.limit_list.append(time_second)
            if len(self.limit_list) > self.store_length:
                self.limit_list.pop(0)

        string += "    " * tabs
        string += "{}".format(item)
        if avg_window != -1 and len(self.limit_list) >= 2:
            string += "   \t{0:.{3}f}s from last print, {1}-step avg: {2:.{3}f}s".format(
                self.limit_list[-1] - self.limit_list[-2], avg_window,
                (self.limit_list[-1] - self.limit_list[-min(avg_window+1,len(self.limit_list))]) / avg_window,
                self.n_digits,
            )

        if banner_size > 0:
            print("=" * banner_size)
        print(string, end=end)
        if banner_size > 0:
            print("=" * banner_size)
        try:
            sys.stdout.flush()
        except:
            pass

    def warning(self, item):
        print(colored(item, 'yellow'))
        try:
            sys.stdout.flush()
        except:
            pass

    def error(self, item):
        raise Exception("{}".format(item))
def get_time(is_bracket=True, return_numerical_time=False, precision="second"):
    """Get the string of the current local time."""
    from time import localtime, strftime, time
    if precision == "second":
        string = strftime("%Y-%m-%d %H:%M:%S", localtime())
    elif precision == "millisecond":
        string = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    if is_bracket:
        string = "[{}] ".format(string)
    if return_numerical_time:
        return string, time()
    else:
        return string
p = Printer(n_digits=6)

# Config utils for tyro
def get_config_hash(config: Any) -> str:
    """Generate a hash string from configuration object.
    
    Args:
        config: Configuration object (dataclass instance)
        
    Returns:
        SHA-256 hash string of the configuration
    """
    # Convert dataclass to dictionary recursively
    if hasattr(config, '__dict__'):
        config_dict = {}
        for key, value in config.__dict__.items():
            if hasattr(value, '__dict__'):  # Handle nested dataclasses
                config_dict[key] = asdict(value)
            else:
                config_dict[key] = value
    else:
        config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else vars(config)
    
    # Create deterministic JSON string and hash it
    config_json = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(config_json.encode()).hexdigest()


def setup_experiment_directory(config: Any, base_results_path: str) -> Tuple[str, str]:
    """Set up experiment directory with proper naming convention.
    
    Args:
        config: Training or evaluation configuration object
        base_results_path: Base path for results directory
        
    Returns:
        Tuple of (experiment_directory_path, config_hash_8chars)
        
    Raises:
        OSError: If directory creation fails
    """
    config_hash = get_config_hash(config)[:8]
    
    # Get experiment name and date from config
    if hasattr(config, 'training'):
        exp_name = config.training.exp_name
        date_exp = config.training.date_exp
    elif hasattr(config, 'evaluation'):
        exp_name = config.evaluation.exp_name
        date_exp = config.evaluation.date_exp
    else:
        exp_name = getattr(config, 'exp_name', 'default_exp')
        date_exp = getattr(config, 'date_exp', datetime.now().strftime('%Y-%m-%d'))
    
    # Create experiment directory structure
    exp_dir = os.path.join(base_results_path, date_exp, f"{exp_name}_{config_hash}")
    
    # Create subdirectories
    subdirs = ['checkpoints', 'logs', 'plots', 'inference_results']
    for subdir in subdirs:
        os.makedirs(os.path.join(exp_dir, subdir), exist_ok=True)
    
    return exp_dir, config_hash


def save_config_from_tyro(config: Any, config_dir: str) -> str:
    """Save tyro configuration to YAML file.
    
    Args:
        config: Tyro configuration object (dataclass instance)
        config_dir: Directory to save the config file
        
    Returns:
        Path to the saved configuration file
        
    Raises:
        OSError: If directory creation or file writing fails
    """
    os.makedirs(config_dir, exist_ok=True)
    config_file_path = os.path.join(config_dir, 'config.yaml')
    
    # Convert dataclass to dictionary recursively
    if hasattr(config, '__dict__'):
        config_dict = {}
        for key, value in config.__dict__.items():
            if hasattr(value, '__dict__'):  # Handle nested dataclasses
                config_dict[key] = asdict(value)
            else:
                config_dict[key] = value
    else:
        config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else vars(config)
    
    with open(config_file_path, 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=False, indent=2)
    
    return config_file_path


def load_and_override_config(config_class: type, config_path: Optional[str] = None) -> Any:
    """Load configuration from file and override defaults.
    
    Args:
        config_class: Configuration class to instantiate
        config_path: Path to YAML config file. If None, use default parameters
        
    Returns:
        Configuration object with loaded/default parameters
        
    Raises:
        FileNotFoundError: If config_path is provided but file doesn't exist
        ValueError: If YAML file is invalid or incompatible with config class
    """
    if config_path is None:
        # Use default parameters
        return config_class()
    
    # Load configuration from file
    config_dict = load_config_from_yaml(config_path)
    
    try:
        # Create config object with loaded parameters
        return config_class(**config_dict)
    except TypeError as e:
        raise ValueError(f"Configuration file incompatible with {config_class.__name__}: {e}")

def load_config_from_yaml(config_path: str) -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML config file
        
    Returns:
        Dictionary containing configuration parameters
    """
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {config_path}: {e}")

# Deprecated functions for backward compatibility
def add_args_from_config(parser):
    """Deprecated: Use tyro configuration classes instead."""
    import warnings
    warnings.warn("add_args_from_config is deprecated. Use tyro configuration classes instead.", 
                  DeprecationWarning, stacklevel=2)
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    existing_args = {action.dest for action in parser._actions}
    
    # Add arguments from the config file to the parser
    for key, value in config.items():
        # If the argument is not already added, add it to the parser
        if key not in existing_args:
            parser.add_argument(f'--{key}', type=type(value), default=value)
    
    return parser

def save_config_from_args(args):
    """Deprecated: Use save_config_from_tyro instead."""
    import warnings
    warnings.warn("save_config_from_args is deprecated. Use save_config_from_tyro instead.", 
                  DeprecationWarning, stacklevel=2)
    config_dict = {k: v for k, v in vars(args).items() if k != 'config'}
    
    config_dir = args.results_path
    os.makedirs(config_dir, exist_ok=True)  # Ensure the directory exists
    config_file_path = os.path.join(config_dir, 'config.yaml')
    with open(config_file_path, 'w') as file:
        yaml.dump(config_dict, file, default_flow_style=False)

    return config_file_path 


# result analysis
def caculate_confidence_interval(data):
    ''''
    input example: abs(pred_design-pred_simu)
    '''
    list_dim=range(data.dim())
    if data.dim()>1:
        MAE_batch_size = torch.mean(data, dim=tuple(list_dim[1:]))
    else:
        MAE_batch_size =data
    mean = torch.mean(MAE_batch_size)
    
    std_dev = torch.std(MAE_batch_size)
    min_value=min(MAE_batch_size)
    confidence_level = 0.95
    # pdb.set_trace()
    n = len(data)
    # kk = stats.t.ppf((1 + confidence_level) / 2, n - 1) * (std_dev / (n ** 0.5))
    margin_of_error= std_dev* 1.96/ torch.sqrt(torch.tensor(n,dtype=float))
    confidence_interval = (mean - margin_of_error, mean + margin_of_error)

    print("mean:", mean.item())
    print("std:", std_dev.item())
    print(f"margin_of_error:", margin_of_error)

    return mean,std_dev,margin_of_error,min_value


#training utils
def caculate_num_parameters(model):
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    # pdb.set_trace()
    return
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return

def draw_loss(training_loss_list: list, test_loss_list: list, save_path: str) -> None:
    """Plot training and testing loss curves and save to file.
    
    Creates a line plot showing the progression of training and testing losses
    over epochs and saves it to the specified path.
    
    Args:
        training_loss_list: List of training loss values, shape [num_epochs]
        test_loss_list: List of testing loss values, shape [num_epochs]
        save_path: Full path where the plot image will be saved (including filename)
        
    Returns:
        None
        
    Raises:
        OSError: If the save directory doesn't exist or is not writable
        ValueError: If loss lists have different lengths or are empty
    """
    if len(training_loss_list) != len(test_loss_list):
        raise ValueError(f"Loss lists must have same length: {len(training_loss_list)} vs {len(test_loss_list)}")
    
    if len(training_loss_list) == 0:
        raise ValueError("Loss lists cannot be empty")
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create figure with proper size and DPI
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    # Plot training and testing loss with distinct colors
    epochs = range(1, len(training_loss_list) + 1)
    ax.plot(epochs, training_loss_list, 'b-', label='Training Loss', linewidth=2)
    ax.plot(epochs, test_loss_list, 'r-', label='Testing Loss', linewidth=2)

    # Set title and labels with proper formatting
    ax.set_title('Training and Testing Loss During Training', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3)
    
    # Display legend with proper positioning
    ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    # Save the plot to the specified file
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Close the figure to free memory
    plt.close(fig)


def get_entropy(X, Y=None, K=3, NN=100, normalize=False, stop_grad_reference=False):
    '''
    Estimating differential entropy with K-nearest neighbors.
    From Lombardi, Damiano, and Sanjay Pant. "Nonparametric k-nearest-neighbor entropy estimator."
        Physical Review E 93.1 (2016): 013310.

    Args:
        X, Y: torch.tensor, with shape of [N, F] or [N], where N is the number of samples,
            and F is the feature size. If Y is not None, then compute the entropy of H(X, Y).
        K: K^th nearest neighbor. Slightly larger K yields more stable estimates.
        NN: number of samples to sample from to compute the entropy.
        normalize: whether to normalize the X (and Y). Default False.
        stop_grad_reference: if True, will stop gradient for the reference samples.

    Returns:
        entropy: scalar, estimated entropy.
    '''
    epsilon = 1e-20
    MAX = 1e10

    if len(X.shape) > 2:
        raise Exception('The shape of X and Y must be [N, F] or [N]. Please fix the shape!')
    device = X.device
    NN = min(NN, X.shape[0]) # number of points to sample from
    K = min(K, NN)
    indices = torch.randperm(X.shape[0])[:NN]
    if X.ndim == 1:
        X = X[indices, None]
    else:
        X = X[indices]
    if normalize:
        X = (X - X.mean(0)) / (X.std(0) + epsilon)

    if Y is not None:
        if Y.ndim == 1:
            Y = Y[indices, None]
        else:
            Y = Y[indices]
        if normalize:
            Y = (Y - Y.mean(0)) / (Y.std(0) + epsilon)
        X = torch.cat((X, Y), dim=1)
    if stop_grad_reference:
        dist_matrix = torch.norm(X[:, None, :] - X[None, :, :].detach(), p=2, dim=-1)  # dist_matrix: [N, N]
    else:
        dist_matrix = torch.norm(X[:, None, :] - X[None, :, :], p=2, dim=-1)  # dist_matrix: [N, N]
    dist_matrix = torch.where(
        torch.eye(dist_matrix.size(0), dtype=bool, device=device),
        torch.tensor(MAX, dtype=dist_matrix.dtype, device=device),
        dist_matrix
    )  # Assign the diagonal elements to be MAX.
    top_k_values = torch.topk(dist_matrix, K, dim=1, largest=False)[0][:, -1] # [N], obtaining the top K'th distance from each row
    return X.shape[-1] * torch.log(top_k_values + epsilon).mean(0) + torch.digamma(torch.tensor(NN, device=device)) \
            - torch.digamma(torch.tensor(K, device=device)) + torch.log(torch.tensor(torch.pi, device=device)) * X.shape[-1] / 2 \
            - torch.lgamma(torch.tensor(1 + X.shape[-1] / 2, device=device))


def get_mi(X, Y, K=3, NN=100, normalize=False, stop_grad_reference=False):
    '''
    Compute estimation for mutual information.

    Args:
        X, Y: torch.tensor, with shape of [N, F] or [N], where N is the number of samples,
            and F is the feature size. If Y is not None, then compute the entropy of H(X, Y).
        K: K^th nearest neighbor. Slightly larger K yields more stable estimates.
        NN: number of samples to sample from to compute the entropy.
        normalize: whether to normalize the X (and Y). Default False.
        stop_grad_reference: if True, will stop gradient for the reference samples.

    Returns:
        mi: scalar, estimated mutual information.
    '''
    ent_X = entropy(X, K=K, NN=NN, normalize=normalize, stop_grad_reference=stop_grad_reference)
    ent_Y = entropy(Y, K=K, NN=NN, normalize=normalize, stop_grad_reference=stop_grad_reference)
    ent_XY = entropy(X, Y, K=K, NN=NN, normalize=normalize, stop_grad_reference=stop_grad_reference)
    # print(ent_X, ent_Y, ent_XY)
    mutual_information = ent_X + ent_Y - ent_XY
    return mutual_information


def get_hashing(string_repr, length=None):
    """Get the hashing of a string."""
    import hashlib, base64
    hashing = base64.b64encode(hashlib.md5(string_repr.encode('utf-8')).digest()).decode().replace("/", "a")[:-2]
    if length is not None:
        hashing = hashing[:length]
    return hashing
