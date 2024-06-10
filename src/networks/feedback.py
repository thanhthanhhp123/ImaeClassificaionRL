import torch
import torch.nn as nn
from torchvision.ops import Permute

class Feedback(nn.Module):
    def __init__(self, n, n_m, hidden_size) -> None:
        super().__init__()
        
        self.__seq = nn.Sequential(
            nn.Linear(n, hidden_size),
            nn.GELU(),
            Permute([1, 2, 0]),
            nn.BatchNorm1d(hidden_size),
            Permute([2, 0, 1]),
            nn.Linear(hidden_size, n_m),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__seq(x)

class Receiver(nn.Module):
    def __init__(self, n_m, n) -> None:
        super().__init__()
        self.__seq = nn.Sequential(
            nn.Linear(n_m, n),
            nn.GELU(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__seq(x)
