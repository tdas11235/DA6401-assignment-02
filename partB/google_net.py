import torch
import torch.nn as nn
from torchvision import models


def get_googlenet_shallow(n_classes=10):
    model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)

    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False


    # adding an extra layer for 10 classes
    model.fc = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, n_classes)
    )


    # unfreeze the last 2 layers
    for param in model.fc.parameters():
        param.requires_grad = True
    
    return model


def get_googlenet(n_classes=10):
    model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)

    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False


    # adding an extra layer for 10 classes
    model.fc = nn.Sequential(
        nn.Linear(1024, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, n_classes)
    )


    # unfreeze the last 2 blocks (classifier and high level features)
    for name, param in model.named_parameters():
        if "inception5" in name or "fc" in name:
            param.requires_grad = True
    
    return model
