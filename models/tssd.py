import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RSM1D(nn.Module):
    def __init__(self, channels_in=None, channels_out=None, **kwargs):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.conv1 = nn.Conv1d(
            in_channels=channels_in,
            out_channels=channels_out,
            bias=False,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=channels_out,
            out_channels=channels_out,
            bias=False,
            kernel_size=3,
            padding=1,
        )
        self.conv3 = nn.Conv1d(
            in_channels=channels_out,
            out_channels=channels_out,
            bias=False,
            kernel_size=3,
            padding=1,
        )

        self.bn1 = nn.BatchNorm1d(channels_out)
        self.bn2 = nn.BatchNorm1d(channels_out)
        self.bn3 = nn.BatchNorm1d(channels_out)

        self.nin = nn.Conv1d(
            in_channels=channels_in,
            out_channels=channels_out,
            bias=False,
            kernel_size=1,
        )

    def forward(self, xx):
        yy = F.relu(self.bn1(self.conv1(xx)))
        yy = F.relu(self.bn2(self.conv2(yy)))
        yy = self.conv3(yy)
        xx = self.nin(xx)

        xx = self.bn3(xx + yy)
        xx = F.relu(xx)
        return xx


class TSSD(nn.Module):  # Res-TSSDNet
    def __init__(self, in_dim, **kwargs):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=1, out_channels=16, kernel_size=7, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(16)

        self.RSM1 = RSM1D(channels_in=16, channels_out=32)
        self.RSM2 = RSM1D(channels_in=32, channels_out=64)
        self.RSM3 = RSM1D(channels_in=64, channels_out=128)
        self.RSM4 = RSM1D(channels_in=128, channels_out=128)

        self.fc1 = nn.Linear(in_features=128, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.out = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=4)

        # stacked ResNet-Style Modules
        x = self.RSM1(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM2(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM3(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM4(x)
        x = F.max_pool1d(x, kernel_size=x.shape[-1])

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x


if __name__ == "__main__":
    model = TSSD(in_dim=64600)
    x = torch.Tensor(np.random.rand(8, 64600))
    y = model(x)
    print(y.shape)
    print(y)
