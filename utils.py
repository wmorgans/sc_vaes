
from torch import nn
from .types import *


def FC_block(first_layer: int,
             last_layer: int,
             hidden_dims: List):
    modules = []

    # Build Encoder
    for h_dim in hidden_dims:
        modules.append(
            nn.Sequential(
                nn.Linear(first_layer, out_channels=h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU())
        )
        in_channels = h_dim

    modules.append(
        nn.Sequential(
                nn.Linear(in_channels, out_channels=last_layer),
                nn.BatchNorm1d(last_layer),
                nn.ReLU()))

    return nn.Sequential(*modules)