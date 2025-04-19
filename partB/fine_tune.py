import wandb
import os
import sys
import yaml
import torch
import torch.nn as nn
import gc

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from trainer import Trainer
from google_net import get_googlenet


SEED = 42
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def finetune_func():
    model = get_googlenet(n_classes=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)
    loss = nn.CrossEntropyLoss()
    trainer = Trainer(
        data_dir='dataset/inaturalist_12K/',
        model=model,
        optimizer=optimizer,
        loss=loss,
        batch_size=64,
        augment=True,
        best_model_path='partB/models',
        device=DEVICE
    )
    run_name = "googlenet_2"
    trainer.train(model_name=run_name, epochs=5)
    del model
    del optimizer
    del trainer
    torch.cuda.empty_cache()
    gc.collect()


def main():
    project = "da6401-a2-finetune-1"
    wandb.init(project=project)
    finetune_func()


if __name__ == "__main__":
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    main()

