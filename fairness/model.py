import torch.nn as nn


class FullyConnected(nn.Module):

    def __init__(self, dim, **kwargs):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(dim, 60),
            nn.ReLU(),
            nn.Linear(60, 25),
            nn.ReLU(),
            nn.Linear(25, 1),
        )

    def forward(self, x):
        return dict(logits=self.f(x))
