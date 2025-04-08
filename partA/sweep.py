import wandb
import os
import sys
import yaml

project_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_root)

from partA.trainer import Trainer
from partA.cnn import CNN

def load_config(path="partA/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def sweep_func():
    wandb.init()
    config = wandb.config
    
