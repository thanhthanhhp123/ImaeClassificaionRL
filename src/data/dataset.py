import glob
import pickle as pkl
import os

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, io
import tqdm
from PIL import Image
from torchvision.datasets import ImageFolder


class MNIST(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train
        self.data, self.targets = torch.load(
            os.path.join(self.root, 'processed', 'training.pt')
            if self.train
            else os.path.join(self.root, 'processed', 'test.pt')
        )
        self.data = self.data.unsqueeze(1).float()
        self.targets = self.targets.long()

    def __getitem__(self, index):
        x, y = self.data[index], self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)