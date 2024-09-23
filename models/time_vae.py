import torch
from .base_vae import BaseVAE
from torch import nn
from torch import optim
from torch.nn import functional as F
from ..types import *
from ..utils import FC_block
from .rna_vae import VanillaVAE, Gen_rna_vae
import numpy as np

EPS = 1e-7


class Time_Vae(VanillaVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 y_stars: torch.Tensor,
                 edges: np.array,
                 hidden_dims: List | None = [512, 256, 128, 64, 32],
                 encoder: nn.Sequential | None = None,
                 decoder: nn.Sequential | None = None,
                 kl_weight = 0.00025,
                 time_weight = 0.05) -> None:
                
                super().__init__(in_channels, latent_dim, hidden_dims=hidden_dims,
                         encoder=encoder, decoder=decoder, kl_weight=kl_weight)
                
                self.time_weight = time_weight
                self.y_stars = y_stars
                self.edges = edges
                self.time_regressor = RegressorLinear(self.latent_dim)

    def loss_function(self, recons, input, mu, log_var, pred_time, time, **kwargs) -> dict:
        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        ord_loss = self.ord_loss(pred_time, time)

        loss = recons_loss + self.kl_weight * kld_loss + ord_loss*self.time_weight
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()} 
    
    def forward(self, input: Tensor):
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        pred_time = self.time_regressor(z)
        return [self.decode(z), input, mu, log_var, pred_time]
    
    def training_step(self, batch, batch_idx):
        x  =  batch[0]
        time = batch[1]['time_levels']

        losses = self.loss_function(*self.forward(x), time)
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        return losses
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x = batch[0]
        time = batch[1]['time_levels']
        losses = self.loss_function(*self.forward(x), time)
        self.log_dict(losses)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x = batch[0]
        time = batch[1]['time_levels']
        losses = self.loss_function(*self.forward(x), time)

        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        val_loss = losses['loss']
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

                

    def ord_loss(self, pred_ys: torch.Tensor, ys: torch.Tensor, current_epoch=None,
                 max_epoch=None, scale=0.3, start_scale=None) -> torch.Tensor:
        
        #burnin_scale = start_scale*(0.1*max_epoch - current_epoch)/(0.1*max_epoch)  \
        #    if (self.align_burnin and (0.1*max_epoch > current_epoch)) else 0
        #scale = burnin_scale + scale

        std_normal = torch.distributions.normal.Normal(0, scale)

        log_likelihood = torch.tensor(0, dtype=torch.float, requires_grad=True)

        for y, y_pred in zip(ys, pred_ys):
            for i, y_star in enumerate(self.y_stars):
                if y_star[y] == 0:
                    continue
                else:
                    log_likelihood = log_likelihood - \
                        y_star[y]*torch.log(
                          std_normal.cdf(self.edges[i][1] - y_pred)
                          - std_normal.cdf(self.edges[i][0] - y_pred) + EPS)
                                        
        log_likelihood = log_likelihood/len(ys)
                                        
            # log_likelihood = log_likelihood - \
            #                     torch.sum(  
            #                       torch.tensor([y_star[y]*torch.log(std_normal.cdf(self.edges[i][1] - y_pred) -
            #                         std_normal.cdf(self.edges[i][0] - y_pred))
            #                         for i, y_star in enumerate(self.y_stars)]))

        return log_likelihood[0]
    

class RegressorLinear(torch.nn.Module):

    r"""
    Modality discriminator

    Parameters
    ----------
    in_features
        Input dimensionality
    out_features
        Output dimensionality
    """

    def __init__(self, inputSize, outputSize=1):
        super(RegressorLinear, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize, bias=False)

    def forward(self, x):
        out = self.linear(x)
        return out