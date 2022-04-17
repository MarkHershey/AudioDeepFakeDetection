import numpy as np
import torch
import torch.nn as nn
from torch import flatten
from torch.nn import functional as F


class ShallowCNN(nn.Module):
    def __init__(self, in_features, out_dim, **kwargs):
        super(ShallowCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_features, 32, kernel_size=4, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=1)
        self.conv3 = nn.Conv2d(48, 64, kernel_size=4, stride=1, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=(2, 4), stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(15104, 128)
        self.fc2 = nn.Linear(128, out_dim)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    model = ShallowCNN(in_features=1, out_dim=1)
    x = torch.Tensor(np.random.rand(8, 40, 972))
    y = model(x)
    print(y.shape)
    print(y)
