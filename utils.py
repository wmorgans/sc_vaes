
from torch import nn
from .types import *


def FC_block(first_layer: int,
             last_layer: int,
             hidden_dims: List,
             activations: List | None = None):
    modules = []

    if activations is None:
        activations = len(hidden_dims + 1)*[nn.ReLU()]

    # Build Encoder
    for i, h_dim in enumerate(hidden_dims):
        modules.append(
            nn.Sequential(
                nn.Linear(first_layer, out_channels=h_dim),
                nn.BatchNorm1d(h_dim),
                activations[i])
        )
        in_channels = h_dim

    modules.append(
        nn.Sequential(
                nn.Linear(in_channels, out_channels=last_layer),
                nn.BatchNorm1d(last_layer),
                activations[-1]))

    return nn.Sequential(*modules)