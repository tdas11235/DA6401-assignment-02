import torch
import torch.nn as nn
from typing import List, Callable, Tuple, Optional


class CNN(nn.Module):
    
    def __init__(self,
                 conv_filters: List[int],
                 conv_kernels: List[int],
                 conv_activations: List[Callable],
                 fc_layers: List[int],
                 fc_activations: List[Callable],
                 input_shape: Tuple[int, ...] = (3, 224, 224),
                 num_classes: int = 10,
                 use_batch_norm: bool = False,
                 dropout_rate: Optional[float] = None) -> None:
        super(CNN, self).__init__()
        # depth
        in_channels = input_shape[0]
        # image height and width
        H, W = input_shape[1], input_shape[2]
        conv_blocks = []

        # Convolutional layers
        for out_channels, kernel_size, activation in zip(conv_filters, conv_kernels, conv_activations):
            conv_blocks.append(nn.Conv2d(in_channels, out_channels, 
                                         kernel_size=kernel_size, padding=kernel_size//2))
            if use_batch_norm:
                conv_blocks.append(nn.BatchNorm2d(out_channels))
            conv_blocks.append(activation())
            conv_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = out_channels
            H //= 2
            W //= 2
        self.conv_blocks = nn.Sequential(*conv_blocks)
        
        # Fully connected layers
        self.flatten_dim = in_channels * H * W
        fc_blocks = []
        in_features = self.flatten_dim
        for neurons, activation in zip(fc_layers, fc_activations):
            fc_blocks.append(nn.Linear(in_features, neurons))
            fc_blocks.append(activation())
            if dropout_rate is not None:
                fc_blocks.append(nn.Dropout(dropout_rate))
            in_features = neurons
        self.fc_blocks = nn.Sequential(*fc_blocks)

        # Output layer
        self.output_layer = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)
        x = self.fc_blocks(x)
        x = self.output_layer(x)
        return x
        

