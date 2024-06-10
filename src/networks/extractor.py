from abc import ABC, abstractmethod
import torch
from typing import cast
import torch.nn as nn
from torchvision.ops import Permute

class Extractor(nn.Module, ABC):
    @property
    @abstractmethod
    def output_size(self) -> int:
        pass


class CNN(Extractor):
    def __init__(self, f: int):
        super().__init__()

        self._seq = nn.Sequential(
             nn.Conv2d(1, 16, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, (3, 3), padding=1),
            nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(32),
            nn.Flatten(1, -1),
        )

        self.__out_size = 32 * (f // 4) ** 2
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, 0, None, : ,:]
        return self._seq(x)
    
    @property
    def output_size(self) -> int:
        return self.__out_size

class StatetoFeatures(nn.Module):
    def __init__(self, d, n_d):

        self.__seq = nn.Sequential(
            nn.Linear(d, n_d),
            nn.GELU(),
            Permute([1, 2, 0]),
            nn.BatchNorm1d(n_d),
            Permute([2, 0, 1]),
        )

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__seq(x)
