import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from typing import Tuple
from tqdm import trange
import wandb


class Trainer:
    def __init__(self,
                 data_dir: str,
                 model: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 loss: nn.Module,
                 input_size: Tuple[int, ...] = (224, 224),
                 batch_size: int = 64,
                 augment: bool = True,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 best_model_path: str = 'models'
                 ):
        self.device = device
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss = loss
        # taken from ImageNet (assumed to be similar for iNaturalist too)
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]
        # transform for augmentation or otherwise
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ]) if augment else transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        eval_transform = transforms.Compose([
            transforms.Resize(input_size[0] + 32),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
        # loaders for splits of dataset
        self.train_loader = DataLoader(
            datasets.ImageFolder(os.path.join(
                data_dir, 'train'), transform=train_transform),
            batch_size=batch_size,
            shuffle=True
        )
        self.val_loader = DataLoader(
            datasets.ImageFolder(os.path.join(
                data_dir, 'val'), transform=eval_transform),
            batch_size=batch_size,
            shuffle=False
        )
        self.test_loader = DataLoader(
            datasets.ImageFolder(os.path.join(
                data_dir, 'test'), transform=eval_transform),
            batch_size=batch_size,
            shuffle=False
        )
        self.model_folder = best_model_path
        os.makedirs(self.model_folder, exist_ok=True)

    def train_epoch(self):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
        acc = correct / total
        return total_loss / len(self.train_loader), acc
    
    def evaluate(self, split: str = 'val'):
        self.model.eval()
        loader = {
            'val': self.val_loader,
            'test': self.test_loader
        }.get(split, self.val_loader)
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        return total_loss / len(loader), acc
    
    def train(self, model_name: str, epochs: int = 10):
        best_val_acc = 0.0
        epoch_bar = trange(epochs, desc="Training", unit="epoch")
        for epoch in epoch_bar:
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate(split='val')
            epoch_bar.set_postfix({
                "Train Loss": f"{train_loss:.4f}",
                "Train Acc": f"{train_acc:.3f}",
                "Val Loss": f"{val_loss:.4f}",
                "Val Acc": f"{val_acc:.3f}"
            })
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            })
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = f"{self.model_folder}/{model_name}_best.pth"
                torch.save(self.model.state_dict(), save_path)
                print(f"New best model saved: {save_path}")