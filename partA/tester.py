import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from typing import Tuple
import wandb
import matplotlib.pyplot as plt


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

    def show_grid(self):
        self.model.eval()
        samples = []
        class_names = self.test_loader.dataset.classes
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)

                for img, pred, label in zip(images, preds, labels):
                    samples.append((img.cpu(), pred.item(), label.item()))
                    if len(samples) >= 10:
                        break
                if len(samples) >= 10:
                    break
        fig, axes = plt.subplots(nrows=10, ncols=3, figsize=(12, 30))
        fig.suptitle("Predictions on Test Images", fontsize=20)
        for i, (img, pred, label) in enumerate(samples):
            img = img.permute(1, 2, 0).numpy()
            img = img * 0.2 + 0.5
            img = img.clip(0, 1)
            row = i
            axes[row, 0].imshow(img)
            axes[row, 0].axis('off')
            axes[row, 0].set_title("Original", fontsize=10)
            axes[row, 1].imshow(img)
            axes[row, 1].axis('off')
            pred_label = class_names[pred]
            axes[row, 1].set_title(
                f"Prediction:\n{pred_label}", color='green' if pred == label else 'red', fontsize=10)

            axes[row, 2].imshow(img)
            axes[row, 2].axis('off')
            axes[row, 2].set_title(
                f"Ground Truth:\n{class_names[label]}", fontsize=10)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()



