import torch
import torch.nn as nn
from torchvision.ops import Permute

class LSTMCellWrapper(nn.Module):
    def __init__(self, input_size, n):
        super().__init__()

        self.__lstm = nn.LSTMCell(input_size, n)

    
    def forward(self, h, c, u):
        nb_ag, batch_size, _ = h.size()

        h, c, u = (
            h.flatten(0, 1),
            c.flatten(0, 1),
            u.flatten(0, 1),
        )

        h_next, c_next = self.__lstm(u, (h, c))

        return (
            h_next.view(nb_ag, batch_size, -1),
            c_next.view(nb_ag, batch_size, -1),
        )
    

class Prediction(nn.Module):
    def __init__(self, n, nb_class, hidden_size):
        super().__init__()

        self.__seq_lin = nn.Sequential(
            nn.Linear(self.__n, hidden_size),
            nn.GELU(),
            Permute([1, 2, 0]),
            nn.BatchNorm1d(hidden_size),
            Permute([2, 0, 1]),
            nn.Linear(hidden_size, self.__nb_class),
        )

    def forward(self, x) -> torch.Tensor:
        return self.__seq_lin(x)