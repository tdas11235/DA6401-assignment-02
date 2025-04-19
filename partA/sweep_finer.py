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
from cnn import CNN


SEED = 42
COUNT = 30
MAX_CAP = 512
MIN_CAP = 8
CONV_SIZE = 5
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    
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

def get_kernel(start_kernel: int, org: str) -> list:
    pattern = [start_kernel]
    for _ in range(1, CONV_SIZE):
        if org == 'decrease':
            next_kernel = max(3, pattern[-1] - 2)
        elif org == 'same':
            next_kernel = pattern[-1]
        else:
            raise NotImplementedError
        pattern.append(next_kernel)
    return pattern

def sweep_func(config):
    conv_act = get_activation(config.activation_pair[0])
    fc_act = get_activation(config.activation_pair[1])
    conv_filters = get_filters(
        config.start_filters, config.filter_organization)
    conv_kernels = get_kernel(config.start_kernel, config.kernel_organization)
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
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                 weight_decay=config.weight_decay)
    loss = nn.CrossEntropyLoss()
    trainer = Trainer(
        data_dir='dataset/inaturalist_12K/',
        model=model,
        optimizer=optimizer,
        loss=loss,
        batch_size=64,
        augment=config.data_augmentation,
        best_model_path='partA/models_top_5',
        device=DEVICE
    )
    aug_flag = "Aug" if config.data_augmentation else "NoAug"
    bn_flag = "BN" if config.batch_norm else "NoBN"
    do_flag = f"DO{config.dropout}"
    run_name = (
        f"F{config.start_filters}-{config.filter_organization}_"
        f"A{config.activation_pair[0]}-{config.activation_pair[1]}_"
        f"FC{config.fc_neurons}_"
        f"K{config.start_kernel}-{config.kernel_organization}_"
        f"{do_flag}_{bn_flag}_{aug_flag}_LR{config.learning_rate}_"
        f"WD{config.weight_decay}"
    )
    wandb.run.name = run_name
    trainer.train(model_name=run_name, epochs=25)
    del model
    del optimizer
    del trainer
    torch.cuda.empty_cache()
    gc.collect()


def main():
    entity = "ch21b108-indian-institute-of-technology-madras"
    coarse_project = "da6401-a2-test-2"
    sweep_id = "mj2xy5gz"
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{coarse_project}/{sweep_id}")

    # sort runs by validation accuracy in descending order
    top_runs = sorted(
        sweep.runs,
        key=lambda run: run.summary.get("best_val_acc", 0),
        reverse=True
    )[:5]

    project = "da6401-a2-test-2-top-5"
    for run in top_runs:
        config_dict = dict(run.config)
        wandb.init(project=project, config=config_dict, reinit=True)
        sweep_func(wandb.config)
    wandb.finish()



if __name__ == "__main__":
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    main()

