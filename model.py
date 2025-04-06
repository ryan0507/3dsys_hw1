import torch
import torch.nn as nn


class RotationEstimationModel(nn.Module):
    def __init__(self, in_size, out_size):
        """
        Model for rotation estimation from point clouds.

        Args:
            in_size (tuple): Input size as (channels, points)
            out_size (int): Output size (depends on rotation representation)
        """
        super(RotationEstimationModel, self).__init__()

        self.LR = nn.LeakyReLU(0.3)
        self.net = nn.Sequential(
            nn.Conv1d(in_size[0], 64, kernel_size=1),
            self.LR,
            nn.Conv1d(64, 128, kernel_size=1),
            self.LR,
            nn.Conv1d(128, 256, kernel_size=1),
            self.LR,
            nn.Conv1d(256, 1024, kernel_size=1),
            self.LR,
            nn.MaxPool1d(kernel_size=in_size[1]),
            nn.Flatten(),
            nn.Linear(1024, 512),
            nn.Dropout(0.5),
            self.LR,
            nn.Linear(512, 512),
            nn.Dropout(0.3),
            self.LR,
            nn.Linear(512, out_size),
        )

        # Initialize weights
        def init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.0)

        self.net.apply(init_weights)

    def forward(self, x):
        return self.net(x)
