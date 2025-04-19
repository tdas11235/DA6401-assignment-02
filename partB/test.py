import torch
import torch.nn as nn
import os
import sys
import wandb

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
if current_dir not in sys.path:
    sys.path.append(current_dir)

from google_net import get_googlenet
from tester import Tester


SEED = 42
COUNT = 30
MAX_CAP = 512
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


def get_results():
    model = get_googlenet(n_classes=10)
    loss = nn.CrossEntropyLoss()
    tester = Tester(
        data_dir='dataset/inaturalist_12K/',
        model=model,
        loss=loss,
        batch_size=64,
        best_model_path='partB/models/googlenet_2_best.pth',
        device=DEVICE
    )
    run_name = 'google_net'
    wandb.run.name = run_name
    test_loss, acc = tester.evaluate()
    wandb.log({"test_loss": test_loss, "test_acc": acc})
    tester.show_grid()

def main():
    project = "da6401-a2-test-2-top-1-1"
    wandb.init(project=project)
    get_results()
    wandb.finish()


if __name__ == "__main__":
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    main()