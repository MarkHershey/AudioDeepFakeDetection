import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F


class SimpleLSTM(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        time_dim: int,
        mid_dim: int,
        out_dim: int,
        **kwargs,
    ):
        super(SimpleLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=mid_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.01,
        )
        self.conv = nn.Conv1d(in_channels=mid_dim * 2, out_channels=10, kernel_size=1)
        self.fc = nn.Linear(in_features=time_dim * 10, out_features=out_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [Tensor] (batch_size, feat_dim, time_dim)
        return:
            output representation [Tensor] (batch_size, out_dim)
        """
        B = x.size(0)

        x = x.permute(0, 2, 1)  # (batch_size, time_dim, feat_dim)

        lstm_out, _ = self.lstm(x)  # (B, T, C=mid_dim * 2)

        feat = lstm_out.permute(0, 2, 1)  # (B, C=mid_dim * 2, T)

        feat = self.conv(feat)  # (B, C, T)
        feat = F.relu(feat)  # (B, C, T)
        feat = feat.reshape(B, -1)  # (B, C*T)
        out = self.fc(feat)  # (B, out_dim)

        return out


class WaveLSTM(nn.Module):
    def __init__(
        self,
        num_frames: int,
        input_len: int,
        hidden_dim: int,
        out_dim: int,
        **kwargs,
    ):
        super(WaveLSTM, self).__init__()

        self.num_frames = num_frames
        self.num_feats = input_len // num_frames

        self.lstm = nn.LSTM(
            input_size=self.num_feats,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.01,
        )
        self.conv = nn.Conv1d(
            in_channels=hidden_dim * 2, out_channels=10, kernel_size=1
        )
        self.fc = nn.Linear(in_features=self.num_frames * 10, out_features=out_dim)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [Tensor] (batch_size, feat_dim)
        return:
            output representation [Tensor] (batch_size, out_dim)
        """
        B = x.size(0)
        x = x.reshape(B, self.num_frames, self.num_feats)

        lstm_out, _ = self.lstm(x)  # (B, T, C=hidden_dim * 2)

        feat = lstm_out.permute(0, 2, 1)  # (B, C=hidden_dim * 2, T)

        feat = self.conv(feat)  # (B, C, T)
        feat = F.relu(feat)  # (B, C, T)
        feat = feat.reshape(B, -1)  # (B, C*T)
        out = self.fc(feat)  # (B, out_dim)

        return out


if __name__ == "__main__":
    # model = SimpleLSTM(feat_dim=40, time_dim=972, mid_dim=30, out_dim=1)
    # x = torch.Tensor(np.random.rand(8, 40, 972))
    # y = model(x)
    # print(y.shape)
    # print(y)

    model = WaveLSTM(num_frames=10, input_len=64600, hidden_dim=30, out_dim=1)
    x = torch.Tensor(np.random.rand(8, 64600))
    y = model(x)
    print(y.shape)
    print(y)
