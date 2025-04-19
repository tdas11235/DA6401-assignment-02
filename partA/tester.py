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


    
    # def grid_samples(self):
    #     self.model.eval()
    #     correct_samples = {}
    #     incorrect_samples = {}
    #     class_names = class_names = self.test_loader.dataset.classes
        
    #     with torch.no_grad():
    #         for images, labels in self.test_loader:
    #             images = images.to(self.device)
    #             labels = labels.to(self.device)
    #             outputs = self.model(images)
    #             _, preds = torch.max(outputs, 1)
    #             confidences = F.softmax(outputs, dim=1)

    #             for img, pred, label, confidence in zip(images, preds, labels, confidences):
    #                 label_id = label.item()
    #                 pred_id = pred.item()
    #                 img_cpu = img.cpu()
    #                 confidence_score = confidence[pred_id].item()

    #                 if label_id not in correct_samples and pred_id == label_id:
    #                     correct_samples[label_id] = (img_cpu, pred_id, label_id, confidence_score)
    #                 elif label_id not in incorrect_samples and pred_id != label_id:
    #                     incorrect_samples[label_id] = (img_cpu, pred_id, label_id, confidence_score)
    #                 total_classes = set(correct_samples.keys()).union(set(incorrect_samples.keys()))
    #                 if len(total_classes) >= len(class_names):
    #                     break
    #             if len(total_classes) >= len(class_names):
    #                 break
    #     samples = []
    #     used_classes = set()
    #     for cls in incorrect_samples:
    #         if len(samples) >= 5:
    #             break
    #         samples.append(incorrect_samples[cls])
    #         used_classes.add(cls)
    #     for cls in correct_samples:
    #         if len(samples) >= 10:
    #             break
    #         if cls not in used_classes:
    #             samples.append(correct_samples[cls])
    #             used_classes.add(cls)
    #     for source in [incorrect_samples, correct_samples]:
    #         for cls in source:
    #             if len(samples) >= 10:
    #                 break
    #             if cls not in used_classes:
    #                 samples.append(source[cls])
    #                 used_classes.add(cls)

    #     return samples, class_names
    
    # def show_grid(self):
    #     # Get balanced 10 samples
    #     samples, class_names = self.grid_samples()
    #     table = wandb.Table(columns=["Image", "True Label", "Prediction (with confidence)"])
    #     for img, pred, label, conf in samples:
    #         img_np = img.permute(1, 2, 0).numpy()
    #         mean = torch.tensor(self.mean)
    #         std = torch.tensor(self.std)
    #         img_np = (img_np * std.numpy()) + mean.numpy()
    #         img_np = img_np.clip(0, 1)

    #         true_label = class_names[label]
    #         pred_label = class_names[pred]
    #         conf_percent = f"{conf * 100:.2f}%"

    #         # Style prediction text
    #         if pred == label:
    #             pred_display = f"<span style='color:#28a745'>{pred_label} ({conf_percent})</span>"
    #         else:
    #             pred_display = f"<span style='color:#dc3545'>{pred_label} ({conf_percent})</span>"

    #         table.add_data(wandb.Image(img_np), true_label, wandb.Html(pred_display))

    #     wandb.log({"Predictions Table": table})



    # def show_grid(self):
    #     self.model.eval()
    #     samples = []
    #     class_names = self.test_loader.dataset.classes
    #     with torch.no_grad():
    #         for images, labels in self.test_loader:
    #             images = images.to(self.device)
    #             labels = labels.to(self.device)

    #             outputs = self.model(images)
    #             _, preds = torch.max(outputs, 1)

    #             for img, pred, label in zip(images, preds, labels):
    #                 samples.append((img.cpu(), pred.item(), label.item()))
    #                 if len(samples) >= 10:
    #                     break
    #             if len(samples) >= 10:
    #                 break
    #     fig, axes = plt.subplots(nrows=10, ncols=3, figsize=(12, 30))
    #     fig.suptitle("Predictions on Test Images", fontsize=20)
    #     for i, (img, pred, label) in enumerate(samples):
    #         img = img.permute(1, 2, 0).numpy()
    #         img = img * 0.2 + 0.5
    #         img = img.clip(0, 1)
    #         row = i
    #         axes[row, 0].imshow(img)
    #         axes[row, 0].axis('off')
    #         axes[row, 0].set_title("Original", fontsize=10)
    #         axes[row, 1].imshow(img)
    #         axes[row, 1].axis('off')
    #         pred_label = class_names[pred]
    #         axes[row, 1].set_title(
    #             f"Prediction:\n{pred_label}", color='green' if pred == label else 'red', fontsize=10)

    #         axes[row, 2].imshow(img)
    #         axes[row, 2].axis('off')
    #         axes[row, 2].set_title(
    #             f"Ground Truth:\n{class_names[label]}", fontsize=10)

    #     plt.tight_layout(rect=[0, 0, 1, 0.96])
    #     wandb.log({"Prediction Grid": wandb.Image(fig)})
    #     plt.close(fig)



