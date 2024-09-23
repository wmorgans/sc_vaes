
from torch import nn
from .types import *
import pandas as pd
import numpy as np
import torch

def FC_block(first_layer: int,
             last_layer: int,
             hidden_dims: List,
             activations: List | None = None):
    modules = []

    if activations is None:
        activations = (len(hidden_dims) + 1)*[nn.ReLU()]
    
    in_channels = first_layer
    # Build Encoder
    for i, h_dim in enumerate(hidden_dims):
        modules.append(
            nn.Sequential(
                nn.Linear(in_channels, h_dim),
                nn.BatchNorm1d(h_dim),
                activations[i])
        )
        in_channels = h_dim

    modules.append(
        nn.Sequential(
                nn.Linear(in_channels, last_layer),
                nn.BatchNorm1d(last_layer),
                activations[-1]))

    return nn.Sequential(*modules)

def string_to_splits(series: pd.Series):
    '''
    Format should be list of  number_number
    '''
    edges = np.array(list(pd.Index(series).dropna().drop_duplicates().sort_values().str.split('_')), dtype=float)
    num_cats = len(edges)
    
    vals = np.sort(edges.ravel())

    new_edges = []
    i = 0
    for i in range(len(vals)-1):
        if vals[i] == vals[i+1]:
            continue
        new_edges.append((vals[i], vals[i+1]))
    
    new_edges = torch.tensor(new_edges, dtype=torch.float, device='cuda')
    y_stars = torch.zeros((len(new_edges), len(edges)), device='cuda')

    for iy, ix in np.ndindex(y_stars.shape):
        y_stars[iy, ix] = 1 if (torch.mean(new_edges[iy]) > edges[ix][0]) & \
                               (torch.mean(new_edges[iy]) < edges[ix][1]) \
                            else 0
    
    mask = ~torch.all(y_stars == 0, axis=1)
    y_stars = y_stars[mask]  #drop rows of all 0
    new_edges = new_edges[mask]

    y_stars = y_stars/y_stars.sum(axis=1, keepdims=True)
    new_edges[[0, -1], [0, -1]] = torch.tensor([-np.inf, np.inf], device='cuda')

    return y_stars, new_edges

def string_to_splits_np(series: pd.Series):
    '''
    Format should be list of  number_number
    '''
    edges = np.array(list(pd.Index(series).dropna().drop_duplicates().sort_values().str.split('_')), dtype=float)
    num_cats = len(edges)
    
    vals = np.sort(edges.ravel())

    new_edges = []
    i = 0
    for i in range(len(vals)-1):
        if vals[i] == vals[i+1]:
            continue
        new_edges.append((vals[i], vals[i+1]))
    
    new_edges = np.array(new_edges)
    y_stars = np.zeros((len(new_edges), len(edges)))

    for iy, ix in np.ndindex(y_stars.shape):
        y_stars[iy, ix] = 1 if (np.mean(new_edges[iy]) > edges[ix][0]) & \
                               (np.mean(new_edges[iy]) < edges[ix][1]) \
                            else 0
    
    mask = ~np.all(y_stars == 0, axis=1)
    y_stars = y_stars[mask]  #drop rows of all 0
    new_edges = new_edges[mask]

    y_stars = y_stars/y_stars.sum(axis=1, keepdims=True)
    new_edges[[0, -1], [0, -1]] = np.array([-np.inf, np.inf])

    return y_stars, new_edges

def str_to_ord_levels(time: pd.Series):
    levels = pd.Index(time).dropna().drop_duplicates().sort_values()  #levels of time
    return levels.get_indexer(time)