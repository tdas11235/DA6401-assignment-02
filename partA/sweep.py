import wandb
import os
import sys
import yaml
import torch
import torch.nn as nn

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from trainer import Trainer
from cnn import CNN


MAX_CAP = 512
MIN_CAP = 8
CONV_SIZE = 5

def load_config(path="partA/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    

def get_filters(start: int, org: str) -> list:
    filters = []
    for i in range(CONV_SIZE):
        if org == 'same':
            filters.append(start)
        elif org == 'double':
            filters.append(min(start * (2 ** i), MAX_CAP))
        elif org == 'half':
            filters.append(max(start // (2 ** i), MIN_CAP))
        else:
            raise NotImplementedError
    return filters

def get_activation(name: str):
    activations = {
        'ReLU': nn.ReLU,
        'GELU': nn.GELU,
        'SiLU': nn.SiLU,
        'Mish': nn.Mish
    }
    return activations[name]

def sweep_func():
    wandb.init()
    config = wandb.config
    conv_act = get_activation(config.activation_pair[0])
    fc_act = get_activation(config.activation_pair[1])
    conv_filters = get_filters(
        config.start_filters, config.filter_organization)
    conv_kernels = [config.kernel_size] * CONV_SIZE
    conv_activations = [conv_act] * CONV_SIZE
    model = CNN(
        conv_filters=conv_filters,
        conv_kernels=conv_kernels,
        conv_activations=conv_activations,
        fc_layers=[config.fc_neurons],
        fc_activations=[fc_act],
        use_batch_norm=config.batch_norm,
        dropout_rate=config.dropout,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss = nn.CrossEntropyLoss()
    trainer = Trainer(
        data_dir='dataset/inaturalist_12K/',
        model=model,
        optimizer=optimizer,
        loss=loss,
        batch_size=64,
        augment=config.data_augmentation,
        best_model_path='models'
    )
    aug_flag = "Aug" if config.data_augmentation else "NoAug"
    bn_flag = "BN" if config.batch_norm else "NoBN"
    do_flag = f"DO{int(config.dropout * 100)}"
    run_name = (
        f"F{config.start_filters}-{config.filter_organization}_"
        f"A{config.activation_pair[0]}-{config.activation_pair[1]}_"
        f"FC{config.fc_neurons}_K{config.kernel_size}_"
        f"{do_flag}_{bn_flag}_{aug_flag}_LR{config.learning_rate}"
    )
    wandb.run.name = run_name
    trainer.train(model_name=run_name, epochs=10)


def main():
    sweep_config = load_config()
    project = "da6401-a2-1"
    sweep_id = wandb.sweep(sweep_config, project=project)
    wandb.agent(sweep_id, sweep_func, count=30)


if __name__ == "__main__":
    main()

