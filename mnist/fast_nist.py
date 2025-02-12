import torch
from torchvision.datasets import MNIST, FashionMNIST

class FastMNIST(MNIST):
    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', None)
        super().__init__(*args, **kwargs)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data = self.data.unsqueeze(1).float().div(255)
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

class FastFMNIST(FashionMNIST):
    def __init__(self, *args, **kwargs):
        device = kwargs.pop('device', None)
        super().__init__(*args, **kwargs)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data = self.data.unsqueeze(1).float().div(255)
        self.data, self.targets = self.data.to(device), self.targets.to(device)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
