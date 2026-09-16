import torch
from .base_vae import BaseVAE
from torch import nn
from torch import optim
from torch.nn import functional as F
from ..types import *
from ..utils import FC_block, smooth_log
from .atac_vae import Poisson_atac_vae
from .time_vae import RegressorLinear, RegressorNonLinear
import numpy as np

EPS = 1e-10

class TimeVaePoissonMixin():
    def forward(self, input: Tensor, batch):
        mu, log_var = self.encode(input, batch)
        z = self.reparameterize(mu, log_var)
        pred_time = self.time_regressor(z)
        pred_proportions = self.decode(z, batch, self.r_p)
        pred_means = pred_proportions * torch.exp(torch.log(input.sum(dim=1, keepdim=True) + 1e-10))
        
        return [pred_means, input, mu, log_var, pred_time]
    
    def training_step(self, batch, batch_idx):
        x  =  batch[0]
        time = batch[1]['time_levels']
        batch_id = batch[1]['batch']

        losses = self.loss_function(*self.forward(x, batch_id), time)
        self.log_dict(losses, on_step=False, on_epoch=True)
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

        self.log_dict(losses, on_step=False, on_epoch=True)
        val_loss = losses['loss']
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        val_unweighted = losses['Reconstruction_Loss'] + losses['KLD'] + losses['reg_loss']
        self.log("val_unweighted", val_unweighted, on_step=False, on_epoch=True)



class Poisson_atac_vae_cumulative_logit(TimeVaePoissonMixin, Poisson_atac_vae):
    def __init__(self,
                y_stars: torch.Tensor,
                edges: torch.Tensor,
                *args,
                time_weight = 0.2,
                linear_time_head = True,
                **kwargs) -> None:
    
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.time_weight = time_weight
        self.register_buffer("y_stars", y_stars)
        self.register_buffer("edges", edges)

        if linear_time_head:
            self.time_regressor = RegressorLinear(self.latent_dim)
        else:
            self.time_regressor = RegressorNonLinear(self.latent_dim)


    def loss_function(self, pred_means, input, mu, log_var, pred_time, time, **kwargs) -> dict:
        recons_loss = self.get_poisson_loss(input, pred_means)

        # kl loss batch mean -> per element to fit with MSE (and later time regression)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss/mu.shape[1]

        ord_loss = self.cumulative_logit(pred_time, time)

        time_weight = self.time_weight

        loss = self.recon_weight * recons_loss + self.kl_weight * kld_loss + ord_loss*time_weight

    
        return {'loss': loss, 'Reconstruction_Loss':recons_loss,
                'KLD':kld_loss, 'reg_loss':ord_loss}


    def cumulative_logit(self, pred_ys: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        
        pred_ys = pred_ys # (batch, 1)
        edges = self.edges.unsqueeze(0)  # (1, K+1)
        ys = ys.reshape(-1)

        if (self.edges.shape[0] != self.y_stars.shape[1] + 1) or (len(self.edges.shape) != 1):
            raise ValueError('edges should be 1d and length must be number of (new) classes + 1')

        cum_probs = torch.sigmoid(edges - pred_ys)  # (batch, K+1)

        # Convert cumulative probs to class probabilities
        probs = torch.zeros(ys.shape[0], self.y_stars.shape[0], device=pred_ys.device)

        for k in range(0, self.y_stars.shape[0]):
            probs[:, k] = cum_probs[:, k+1] - cum_probs[:, k]
        

        # Step 1: get the predicted probability for the correct class
        
        true_class_probs = (probs * self.y_stars[:, ys].T).sum(dim=1)

        if true_class_probs.isnan().any():
            raise ValueError('NaN values in true class probabilities')
        if true_class_probs.shape != ys.shape:
            raise ValueError('Shape mismatch in true class probabilities. Shape is {}, expected {}'.format(true_class_probs.shape, ys.shape))

        # Step 2: take the negative log-likelihood
        nll = -torch.log(true_class_probs + EPS)

        return nll.mean()

class Poisson_atac_vae_continuation_ratio(TimeVaePoissonMixin, Poisson_atac_vae):
    def __init__(self,
                y_stars: torch.Tensor,
                edges: torch.Tensor,
                *args,
                time_weight = 0.2,
                time_noise = 0.3,
                linear_time_head = True,
                **kwargs) -> None:
    
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.time_weight = time_weight
        self.register_buffer("y_stars", y_stars)
        self.register_buffer("edges", edges)
        self.time_noise = time_noise

        if linear_time_head:
            self.time_regressor = RegressorLinear(self.latent_dim)
        else:
            self.time_regressor = RegressorNonLinear(self.latent_dim)


    def loss_function(self, pred_means, input, mu, log_var, pred_time, time, **kwargs) -> dict:
        recons_loss = self.get_poisson_loss(input, pred_means)

        # kl loss batch mean -> per element to fit with MSE (and later time regression)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss/mu.shape[1]

        ord_loss = self.continuation_ratio(pred_time, time)

        time_weight = self.time_weight

        loss = self.recon_weight * recons_loss + self.kl_weight * kld_loss + ord_loss*time_weight

    
        return {'loss': loss, 'Reconstruction_Loss':recons_loss,
                'KLD':kld_loss, 'reg_loss':ord_loss}
    

    def continuation_ratio(self, pred_ys: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
        
        pred_ys = pred_ys # (batch, 1)
        edges = self.edges.unsqueeze(0)  # (1, K+1)

        conditional_probs = torch.sigmoid(edges - pred_ys)  # (batch, K+1)
        continuation_probability = 1 - conditional_probs
        cumprod_continuation = torch.cumprod(continuation_probability, dim=1)


        probs = torch.cat(
            (
                conditional_probs[:, :1],
                conditional_probs[:, 1:] * cumprod_continuation[:, :-1],
                cumprod_continuation[:, -1:],
            ),
            dim=1,
        )

        true_class_probs = (probs * self.y_stars[:, ys].T).sum(dim=1)

        # Step 2: take the negative log-likelihood
        nll = -torch.log(true_class_probs + EPS)

        return nll.mean()

class Poisson_atac_vae_ord(TimeVaePoissonMixin, Poisson_atac_vae):
    def __init__(self,
                y_stars: torch.Tensor,
                edges: torch.Tensor,
                *args,
                time_weight = 0.2,
                linear_time_head = True,
                time_noise = 0.3,
                **kwargs) -> None:
    
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.time_weight = time_weight
        self.register_buffer("y_stars", y_stars)
        self.register_buffer("edges", edges)
        self.time_noise = time_noise

        if linear_time_head:
            self.time_regressor = RegressorLinear(self.latent_dim)
        else:
            self.time_regressor = RegressorNonLinear(self.latent_dim)


    def loss_function(self, pred_means, input, mu, log_var, pred_time, time, **kwargs) -> dict:
        recons_loss = self.get_poisson_loss(input, pred_means)

        # kl loss batch mean -> per element to fit with MSE (and later time regression)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss/mu.shape[1]

        ord_loss = self.ord_loss(pred_time, time, scale=self.time_noise)

        time_weight = self.time_weight

        loss = self.recon_weight * recons_loss + self.kl_weight * kld_loss + ord_loss*time_weight
    
        return {'loss': loss, 'Reconstruction_Loss':recons_loss,
                'KLD':kld_loss, 'reg_loss':ord_loss}

    def ord_loss(self, pred_ys: torch.Tensor, ys: torch.Tensor, scale=0.3) -> torch.Tensor:
        
        std_normal = torch.distributions.normal.Normal(0, scale)

        # Readable loop version:
        # log_likelihood = torch.tensor(0, dtype=torch.float, requires_grad=True)
        # for y, y_pred in zip(ys, pred_ys):
        #     for i, y_star in enumerate(self.y_stars):
        #         if y_star[y] == 0:
        #             continue
        #         else:
        #             log_likelihood = log_likelihood - \
        #                 y_star[y] * smooth_log(
        #                     std_normal.cdf(self.edges[i][1] - y_pred)
        #                     - std_normal.cdf(self.edges[i][0] - y_pred))

        interval_upper = self.edges[:, 1].unsqueeze(0)
        interval_lower = self.edges[:, 0].unsqueeze(0)
        pred_ys = pred_ys.reshape(-1, 1)
        interval_probs = (
            std_normal.cdf(interval_upper - pred_ys)
            - std_normal.cdf(interval_lower - pred_ys)
        )
        class_weights = self.y_stars[:, ys].T
        log_likelihood = -(
            class_weights * smooth_log(interval_probs)
        ).sum(dim=1).mean()
                                                                                
                                                                                
            # log_likelihood = log_likelihood - \
            #                     torch.sum(  
            #                       torch.tensor([y_star[y]*torch.log(std_normal.cdf(self.edges[i][1] - y_pred) -
            #                         std_normal.cdf(self.edges[i][0] - y_pred))
            #                         for i, y_star in enumerate(self.y_stars)]))

        return log_likelihood
    
class Poisson_atac_vae_reg(TimeVaePoissonMixin, Poisson_atac_vae):
    def __init__(self,
                *args,
                time_weight = 0.2,
                linear_time_head = True,
                **kwargs) -> None:
    
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.time_weight = time_weight

        if linear_time_head:
            self.time_regressor = RegressorLinear(self.latent_dim)
        else:
            self.time_regressor = RegressorNonLinear(self.latent_dim)


    def loss_function(self, pred_means, input, mu, log_var, pred_time, time, **kwargs) -> dict:
        recons_loss = self.get_poisson_loss(input, pred_means)

        # kl loss batch mean -> per element to fit with MSE (and later time regression)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss/mu.shape[1]

        reg_loss = F.mse_loss(pred_time.squeeze(), time.squeeze())

        time_weight = self.time_weight

        loss = self.recon_weight * recons_loss + self.kl_weight * kld_loss + reg_loss*time_weight
    
        return {'loss': loss, 'Reconstruction_Loss':recons_loss,
                'KLD':kld_loss, 'reg_loss':reg_loss}