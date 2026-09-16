import torch
from .base_vae import BaseVAE
from torch import nn
from torch import optim
from torch.nn import functional as F
from ..types import *
from ..utils import FC_block

class Poisson_atac_vae(BaseVAE):
    '''
    Variational Auto-Encoder model with Poisson likelihood for ATAC-seq data.
    Based on the model described in the paper "https://www.nature.com/articles/s41592-023-02112-6"
    '''
    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 hidden_dims: List | None = None,
                 recon_weight = 1,
                 kl_weight = 0.2,
                 learning_rate = 1e-3,
                 n_batch: int = 1,
                 decoder: nn.Module | None = None,
                 encoder: nn.Module | None = None,
                 **kwargs) -> None:
        super(Poisson_atac_vae, self).__init__()

        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate
        self.n_batch = n_batch
        self.hidden_dims = hidden_dims if hidden_dims is not None else [512, 256, 128, 64, 32]

        # Build Encoder
        if encoder is None:
            encoder = FC_block(input_dim + n_batch, latent_dim, self.hidden_dims)

        self.encoder = encoder

        if decoder is None:
            self.hidden_dims.reverse()
            decoder = FC_block(latent_dim + n_batch, input_dim, self.hidden_dims)
            self.hidden_dims.reverse()

        self.decoder = decoder

        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)
        self.sp = torch.nn.Softmax(dim=1)
        self.r_p = nn.Parameter(torch.ones(1, input_dim), requires_grad=True)  #region specific scaling factor for each feature



    def encode(self, input: Tensor, batch: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network and returns the latent codes.
        # :param inoput: (Tensor) Input tensor to encoder [N x F]
        :param batch: (Tensor) Batch tensor [N x B] With one hot encoding of batch membership
        :return: (Tensor) List of latent means and log variances of the latent distribution
                    [[ N x D], [N x D]]
        """
        input = torch.cat((input, batch), dim=1)
        result = self.encoder(input)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)
        return [mu, log_var]

    def decode(self, z: Tensor, batch: Tensor, r_p) -> Tensor:
        """
        Maps latent variables onto the input space.
        :param z: (Tensor) [N x D]
        :param batch: (Tensor) Batch tensor [N x B] With one hot encoding of batch membership
        :param: r_p: (Tensor) [1 x F] Region-specific bias added to decoder logits
        :param: l_p: (Tensor) [N x 1] Cell specific scaling factor for each cell (log transformed counts)
        :return: (Tensor) [N x F]
        """
        input = torch.cat((z, batch), dim=1)
        result = self.decoder(input)
        pred_proportions = self.sp(result + r_p)

        return pred_proportions

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        std = torch.clamp(std, min=1e-8, max=1e8)
        eps = torch.randn_like(std)
        return mu + (eps * std )

    def forward(self, input: Tensor, batch: Tensor) -> List[Tensor]:
        mu, log_var = self.encode(input, batch)
        z = self.reparameterize(mu, log_var)
        pred_proportions = self.decode(z, batch, self.r_p)
        pred_means = pred_proportions * torch.exp(torch.log(input.sum(dim=1, keepdim=True) + 1e-10))

        return  [input, mu, log_var, pred_means]

    def loss_function(self, *args: Any, **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        input = args[0]
        mu = args[1]
        log_var = args[2]
        pred_means = args[3]

        kld_weight =  self.kl_weight
        recon_weight = self.recon_weight
        recons_loss = self.get_poisson_loss(input, pred_means)

        # kl loss batch mean -> per element to fit with MSE (and later time regression)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        kld_loss = kld_loss/mu.shape[1]

        loss = recon_weight * recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'KLD':kld_loss}


    def get_poisson_loss(self, input: Tensor, pred_mean: Tensor) -> Tensor:
        """
        Computes the Poisson loss function.
        :param input: (Tensor) Input tensor to encoder [N x F]
        :param pred_mean: (Tensor) Predicted mean tensor from decoder [N x F]
        :return: (Tensor) Poisson loss
        """
        recons_loss = F.poisson_nll_loss(
            pred_mean,
            input,
            log_input=False,
            full=True,       # includes log(input!) like Poisson.log_prob
            reduction="none",
        ).sum(dim=-1).mean()

        return recons_loss

    def training_step(self, batch, batch_idx):
        x  =  batch[0]
        batch_id = batch[1]['batch']

        losses = self.loss_function(*self.forward(x, batch_id))
        self.log_dict(losses, on_step=False, on_epoch=True)
        return losses['loss']
    
    def test_step(self, batch, batch_idx):
        x  =  batch[0]
        batch_id = batch[1]['batch']

        losses = self.loss_function(*self.forward(x, batch_id))
        self.log_dict(losses)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x  =  batch[0]
        batch_id = batch[1]['batch']

        losses = self.loss_function(*self.forward(x, batch_id))
        self.log_dict(losses, on_step=False, on_epoch=True)
        val_loss = losses['loss']
        self.log("val_loss", val_loss, on_step=False, on_epoch=True)
        val_unweighted = losses['Reconstruction_Loss'] + losses['KLD']
        self.log("val_unweighted", val_unweighted, on_step=False, on_epoch=True)
    
    def sample(self,
               num_samples:int,
               batch: Tensor | None,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param batch: (Tensor | None) Batch of data
        :param current_device: (Int) Device to run the model
        :return: (Tensor) (proportions as we use read depth to scale the means) [B x F]
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        if batch is None:
            batch = torch.zeros(num_samples, self.n_batch).to(current_device)

        return self.decode(z, batch, self.r_p)
    
    def generate(self, x: Tensor, batch, **kwargs) -> Tensor:
        """
        Given an input cell x, returns the reconstructed image
        :param x: (Tensor) [1 x F]
        :return: (Tensor) [1 x F]
        """

        return self.forward(x, batch)[-1]

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)