from torch import nn, Tensor, flatten
import torch
import torch.nn.functional as F
from typing import Tuple
import numpy as np

KERNEL_SIZE = 3
POOL_SIZE = 2
STRIDE = 1
FINAL_CHANNELS = 16

def randomize_bias(seq: nn.Sequential) -> None:
    for layer in seq:
        if hasattr(layer, 'bias'):
            nn.init.constant_(layer.bias, 0.1)  # Change bias to some desired value, here we use 0.1


class CNN(nn.Module):
    def __init__(self, *args, board_shape: Tuple[int, int], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.board_shape = board_shape
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 8, KERNEL_SIZE),
            nn.Conv2d(8, FINAL_CHANNELS, KERNEL_SIZE, padding=2),
            nn.MaxPool2d((POOL_SIZE, POOL_SIZE), (1, 1)),
            nn.MaxPool2d((POOL_SIZE, POOL_SIZE), (1, 1)),
        )
        
        self.linear_layers = nn.Sequential(
            nn.Linear(self.num_flat_features(n_channels=FINAL_CHANNELS), 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, self.board_shape[1]),
            nn.Softmax(0)
        )

        randomize_bias(self.conv_layers)
        randomize_bias(self.linear_layers)

    def forward(self, x: Tensor):
        player1 = (x * ((x == 1) | (x == 0))).float()
        player2 = (x * ((x == 2) | (x == 0))).float()
        transformed_board = torch.cat((player1, player2), dim=0)

        x = self.conv_layers(transformed_board)
        x = flatten(x)
        x = self.linear_layers(x)

        return x
    
    def _dim_after_conv(self, start_dim: int) -> int:
        after_conv = (start_dim - KERNEL_SIZE) / STRIDE + 1
        after_conv = (after_conv - KERNEL_SIZE + 2 * 2) / STRIDE + 1
        after_conv = (after_conv - POOL_SIZE) / STRIDE + 1
        after_conv = (after_conv - POOL_SIZE) / STRIDE + 1
        return int(after_conv)

    def num_flat_features(self, n_channels: int):
        rows_after_conv = self._dim_after_conv(self.board_shape[0])
        cols_after_conv = self._dim_after_conv(self.board_shape[1])

        return n_channels * rows_after_conv * cols_after_conv  # all dimensions except the batch dimension
