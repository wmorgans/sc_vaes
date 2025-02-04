import torch
from .base_vae import BaseVAE
from torch import nn
from torch import optim
from torch.nn import functional as F
from ..types import *
from ..utils import FC_block
from .rna_vae import VanillaVAE, Gen_rna_vae_nb
import numpy as np

EPS = 1e-7


class Time_Vae_ord(VanillaVAE):

    def __init__(self,
                 y_stars: torch.Tensor,
                 edges: np.array,
                 *args,
                 time_loss_ramp = False,
                 initial_time_weight = 0.01,
                 final_time_weight = 0.5,
                 n_epoch_ramp = 10,
                 time_weight = 0.2,
                 **kwargs) -> None:
                
                super().__init__(*args, **kwargs)

                self.time_loss_ramp = time_loss_ramp
                self.initial_time_weight = initial_time_weight
                self.final_time_weight = final_time_weight
                self.n_epoch_ramp = n_epoch_ramp
                self.y_stars = y_stars
                self.edges = edges
                self.time_weight = time_weight
                self.time_regressor = RegressorLinear(self.latent_dim)

    def get_time_weight(self, epoch):
        if self.time_loss_ramp:
            return self.initial_time_weight + (self.final_time_weight - self.initial_time_weight) * np.min([(epoch / self.n_epoch_ramp),1])
        else:
            return self.time_weight

    def loss_function(self, recons, input, mu, log_var, pred_time, time, **kwargs) -> dict:
        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss / mu.numel()
        
        ord_loss = self.ord_loss(pred_time, time)

        time_weight = self.get_time_weight(self.current_epoch)

        loss = self.recon_weight * recons_loss + self.kl_weight * kld_loss + ord_loss*time_weight
        return {'loss': loss, 'Reconstruction_Loss':recons_loss,
                'KLD':kld_loss, 'ord_loss':ord_loss} 
    
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
        return losses['loss']
    
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
                                                                                
            # log_likelihood = log_likelihood - \
            #                     torch.sum(  
            #                       torch.tensor([y_star[y]*torch.log(std_normal.cdf(self.edges[i][1] - y_pred) -
            #                         std_normal.cdf(self.edges[i][0] - y_pred))
            #                         for i, y_star in enumerate(self.y_stars)]))

        return log_likelihood[0]/pred_ys.numel()


class Time_Vae_reg(Time_Vae_ord, VanillaVAE):
    def __init__(self,
                 *args,
                 time_loss_ramp = False,
                 initial_time_weight = 0.01,
                 final_time_weight = 0.5,
                 n_epoch_ramp = 10,
                 time_weight = 0.2,
                 **kwargs) -> None:
        VanillaVAE.__init__(self, *args, **kwargs)
        self.time_loss_ramp = time_loss_ramp
        self.initial_time_weight = initial_time_weight
        self.final_time_weight = final_time_weight
        self.n_epoch_ramp = n_epoch_ramp
        self.time_weight = time_weight
        self.time_regressor = RegressorLinear(self.latent_dim)
    
    def loss_function(self, recons, input, mu, log_var, pred_time, time, **kwargs) -> dict:
        kld_weight =  self.kl_weight
        time_weight = self.get_time_weight(self.current_epoch)
        recons_loss = F.mse_loss(input, recons)

        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())
        kld_loss = kld_loss/mu.numel()

        reg_loss = F.mse_loss(pred_time, time)

        loss = self.recon_weight * recons_loss + kld_weight * kld_loss + reg_loss*time_weight
        return {'loss': loss, 'Reconstruction_Loss':recons_loss,
                'KLD':kld_loss, 'reg_loss':reg_loss} 

class Time_Vae_reg_non_linear(Time_Vae_reg, VanillaVAE):
    def __init__(self,
        *args,
        time_weight = 0.05,
        time_reg_hidden = [100, 50, 10],
        **kwargs) -> None:
        VanillaVAE.__init__(self, *args, **kwargs)
        self.time_weight = time_weight
        self.time_regressor = FC_block(self.latent_dim, 1, hidden_dims=time_reg_hidden)

class Time_Vae_ord_nb(Gen_rna_vae_nb):
    def __init__(self,
                 y_stars: torch.Tensor,
                 edges: np.array,
                 *args,
                 time_weight = 0.2,
                 **kwargs) -> None:
        
        super().__init__(*args, **kwargs)
        
        self.time_weight = time_weight
        self.y_stars = y_stars
        self.edges = edges
        self.time_regressor = RegressorLinear(self.latent_dim)

    def forward(self, input: Tensor, batch):
        mu, log_var = self.encode(input, batch)
        z = self.reparameterize(mu, log_var)
        pred_time = self.time_regressor(z)
        return [self.decode(z, batch), input, mu, log_var, pred_time]
     
    def loss_function(self, scaled_pred_mean, input, mu, log_var, pred_time, time, **kwargs) -> dict:
        time_weight = self.time_weight
        kld_weight =  self.kl_weight
        recon_weight = self.recon_weight

        recons_loss = self.get_nb_loss(input, scaled_pred_mean)

        kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())
        kld_loss = kld_loss/mu.numel()

        ord_loss = self.ord_loss(pred_time, time)

        loss = recon_weight * recons_loss + kld_weight * kld_loss + ord_loss*time_weight
        return {'loss': loss, 'Reconstruction_Loss':recons_loss,
                'KLD':kld_loss, 'ord_loss':ord_loss}
        
    def training_step(self, batch, batch_idx):
        x  =  batch[0]
        time = batch[1]['time_levels']
        batch_id = batch[1]['batch']

        losses = self.loss_function(*self.forward(x, batch_id), time)
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        return losses['loss']
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x = batch[0]
        time = batch[1]['time_levels']
        batch_id = batch[1]['batch']

        losses = self.loss_function(*self.forward(x, batch_id), time)
        self.log_dict(losses)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x = batch[0]
        time = batch[1]['time_levels']
        batch_id = batch[1]['batch']

        losses = self.loss_function(*self.forward(x, batch_id), time)

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
                                        
        

        return log_likelihood[0]/pred_ys.numel()


class Time_Vae_reg_nb(Time_Vae_ord_nb, Gen_rna_vae_nb):
    def __init__(self,
                *args,
                time_weight = 0.2,
                **kwargs) -> None:
        Gen_rna_vae_nb.__init__(self, *args, **kwargs)
        self.time_weight = time_weight
        self.time_regressor = RegressorLinear(self.latent_dim)
        
    def loss_function(self, scaled_pred_mean, input, mu, log_var, pred_time, time, **kwargs) -> dict:
        recons_loss = self.get_nb_loss(input, scaled_pred_mean)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss/mu.numel()

        reg_loss = F.mse_loss(pred_time, time)
        
        loss = self.recon_weight*recons_loss + self.kl_weight * kld_loss + reg_loss*self.time_weight
        return {'loss': loss, 'Reconstruction_Loss':recons_loss,
                'KLD':kld_loss, 'reg_loss':reg_loss} 
            

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