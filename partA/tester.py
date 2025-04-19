import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from typing import Tuple
import wandb
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Tester:
    def __init__(self,
                 data_dir: str,
                 model: nn.Module,
                 loss: nn.Module,
                 input_size: Tuple[int, ...] = (224, 224),
                 batch_size: int = 64,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 best_model_path: str = 'model.pth'
                 ):
        self.device = device
        self.model = model
        self.loss = loss
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
    
    def evaluate(self):
        self.model.eval()
        loader = self.test_loader
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)
                total_loss += loss.item()
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        return total_loss / len(loader), acc
    
    def collect(self):
        class_names = self.test_loader.dataset.classes
        self.model.eval()
        class_to_examples = {cls: {'tp': None, 'fn': None, 'fp': None} for cls in class_names}
        with torch.no_grad():
            for images, labels in self.test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                confs, preds = torch.max(probs, 1)

                for img, label, pred, conf in zip(images, labels, preds, confs):
                    label_name = class_names[label.item()]
                    pred_name = class_names[pred.item()]
                    if pred == label and class_to_examples[label_name]['tp'] is None:
                        class_to_examples[label_name]['tp'] = (img.cpu(), label_name, pred_name, conf.item())
                    if pred != label and class_to_examples[label_name]['fn'] is None:
                        class_to_examples[label_name]['fn'] = (img.cpu(), label_name, pred_name, conf.item())
                    if pred != label and class_to_examples[pred_name]['fp'] is None:
                        class_to_examples[pred_name]['fp'] = (img.cpu(), label_name, pred_name, conf.item())

                if all(all(v is not None for v in d.values()) for d in class_to_examples.values()):
                    break

        return class_to_examples
    
    def show_grid(self):
        class_names = self.test_loader.dataset.classes
        class_to_examples = self.collect()
        table = wandb.Table(columns=["Correct Prediction", "False Negative", "False Positive"])
        for cls in class_names:
            row = []
            for key in ['tp', 'fn', 'fp']:
                item = class_to_examples[cls][key]
                if item is None:
                    row.append("N/A")
                    continue
                img, true_label, pred_label, conf = item
                img_np = img.permute(1, 2, 0).numpy()
                mean = torch.tensor(self.mean)
                std = torch.tensor(self.std)
                img_np = (img_np * std.numpy()) + mean.numpy()
                img_np = img_np.clip(0, 1)
                correct = (true_label == pred_label)
                caption = self.format_caption(true_label, pred_label, conf, correct)

                row.append(wandb.Image(img_np, caption=caption))

            table.add_data(*row)

        wandb.log({"Prediction Grid": table})

    def format_caption(self, true_label, pred_label, conf, correct):
        emoji = "✅" if correct else "❌"
        return f"True: {true_label} | Pred: {pred_label} ({conf * 100:.2f}%) {emoji}"