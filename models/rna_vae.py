import torch
from .base_vae import BaseVAE
from torch import nn
from torch import optim
from torch.nn import functional as F
from ..types import *
from ..utils import FC_block


class VanillaVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List | None = [512, 256, 128, 64, 32],
                 encoder: nn.Sequential | None = None,
                 decoder: nn.Sequential | None = None,
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        # Build Encoder
        if encoder is None:
            encoder = FC_block(in_channels, latent_dim, hidden_dims)

        self.encoder = encoder

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        if decoder is None:
           hidden_dims.reverse()
           decoder = FC_block(self.latent_dim, in_channels, hidden_dims)

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x F]
        :return: (Tensor) List of latent codes [[N x D], [N x D]]
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x F]
        """
        result = self.decoder(z)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input cell x, returns the reconstructed image
        :param x: (Tensor) [1 x F]
        :return: (Tensor) [1 x F]
        """

        return self.forward(x)[0]
    
    def training_step(self, batch, batch_idx):
        x , _ =  batch
        losses = self.loss_function(self.forward(x))
        return losses
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, _ = batch
        losses = self.loss_function(self.forward(x))
        self.log_dict("test_loss", losses)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x, _ = batch
        losses = self.loss_function(self.forward(x))
        self.log_dict("val_loss", losses)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)