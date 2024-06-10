import torch
import torch.nn as nn
from torchvision.ops import Permute

class Policy(nn.Module):
    def __init__(self, nb_action, n, hidden_size) -> None:
        super().__init__()
        
        self.__seq = nn.Sequential(
            nn.Linear(n, hidden_size),
            nn.GELU(),
            Permute([1, 2, 0]),
            nn.BatchNorm1d(hidden_size),
            Permute([2, 0, 1]),
            nn.Linear(hidden_size, nb_action),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__seq(x)


class Critic(nn.Module):
    def __init__(self, n, hidden_size) -> None:
        super().__init__()
        
        self.__seq = nn.Sequential(
            nn.Linear(n, hidden_size),
            nn.GELU(),
            Permute([1, 2, 0]),
            nn.BatchNorm1d(hidden_size),
            Permute([2, 0, 1]),
            nn.Linear(hidden_size, 1),
            nn.Flatten(-2, -1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__seq(x)