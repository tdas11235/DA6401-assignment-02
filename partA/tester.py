import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from typing import Tuple
import wandb


class Tester:
    def __init__(self,
                 data_dir: str,
                 model: nn.Module,
                 input_size: Tuple[int, ...] = (224, 224),
                 batch_size: int = 64,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 best_model_path: str = 'model.pth'
                 ):
        self.device = device
        self.model = model
        self.model.load_state_dict(torch.load(best_model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        # taken from ImageNet (assumed to be similar for iNaturalist too)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # transform for augmentation or otherwise
        eval_transform = transforms.Compose([
            transforms.Resize(input_size[0] + 32),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        # loaders for splits of dataset
        self.test_loader = DataLoader(
            datasets.ImageFolder(os.path.join(
                data_dir, 'test'), transform=eval_transform),
            batch_size=batch_size,
            shuffle=False
        )

