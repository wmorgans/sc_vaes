import torch
import scanpy as sc
from torch.utils.data import Dataset

class RNA_AnnDataset(Dataset):
    def __init__(self, h5ad_loc, meta_col_names):
        self.file_loc = h5ad_loc
        self.anndata = sc.read_h5ad(self.file_loc)
        self.meta_labs_names = meta_col_names

    def __len__(self):
        return self.anndata.shape[0]
    
    def __getitem__(self, idx):
        cell = self.anndata.X[idx, :]
        meta_labels = {}
        for lab in self.meta_labs_names:
            meta_labels[lab] = self.anndata.obs[idx, lab]
        
        return cell, meta_labels
    
    def __getitems__(self, idxs):
        cells = self.anndata.X[idxs, :]
        meta_labels = {}
        for lab in self.meta_labs_names:
            meta_labels[lab] = self.anndata.obs[idxs, lab]
        return cells, meta_labels