import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=1, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80, out_dim)

    def forward(self, x):
        B = x.size(0)
        x = x.reshape(B, -1)
        y = F.relu(self.fc1(x))
        y = F.sigmoid(self.fc2(y))
        y = self.fc3(y)
        return y


if __name__ == "__main__":
    model = MLP(in_dim=40 * 972, out_dim=1)
    x = torch.Tensor(np.random.rand(8, 40, 972))
    y = model(x)
    print(y.shape)
    print(y)
