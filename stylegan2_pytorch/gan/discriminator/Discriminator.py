from math import log2

from torch import nn
from .DiscriminatorBlock import DiscriminatorBlock


class Discriminator(nn.Module):
    def __init__(self, image_size, network_capacity = 16, transparent = False):
        super().__init__()
        num_layers = int(log2(image_size) - 1)
        num_init_filters = 3 if not transparent else 4

        blocks = []
        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(num_layers + 1)]
        chan_in_out = list(zip(filters[0:-1], filters[1:]))

        blocks = []
        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            is_not_last = ind < (len(chan_in_out) - 1)

            block = DiscriminatorBlock(
                in_chan,
                out_chan,
                downsample = is_not_last
            )
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)
        latent_dim = 2 * 2 * filters[-1]
        self.to_logit = nn.Linear(latent_dim, 1)

    def forward(self, x):
        b, *_ = x.shape
        x = self.blocks(x)
        x = x.reshape(b, -1)
        x = self.to_logit(x)
        return x.squeeze()