from torch import nn
from ...misc import *


class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([nn.Linear(emb, emb), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)