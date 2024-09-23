import torch
from .base_vae import BaseVAE
from torch import nn
from torch import optim
from torch.nn import functional as F
from ..types import *
from ..utils import FC_block

EPS = 1e-8

class VanillaVAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List | None = [512, 256, 128, 64, 32],
                 encoder: nn.Sequential | None = None,
                 decoder: nn.Sequential | None = None,
                 kl_weight = 0.00025) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        # Build Encoder
        if encoder is None:
            encoder = FC_block(in_channels, hidden_dims[-1], hidden_dims[:-2])

        self.encoder = encoder

        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)

        # Build Decoder
        if decoder is None:
           hidden_dims.reverse()
           decoder = FC_block(self.latent_dim, in_channels, hidden_dims)
        
        self.decoder = decoder

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x F]
        :return: (Tensor) List of latent codes [[N x D], [N x D]]
        """
        result = self.encoder(input)
        #result = torch.flatten(result, start_dim=1)

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
        return self.decode(z), input, mu, log_var

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

        kld_weight = self.kl_weight # Account for the minibatch samples from the dataset
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
        x  =  batch[0]
        losses = self.loss_function(*self.forward(x))
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        return losses
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x = batch[0]
        losses = self.loss_function(*self.forward(x))
        self.log_dict(losses)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x = batch[0]
        losses = self.loss_function(*self.forward(x))
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        val_loss = losses['loss']
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)
    
class Gen_rna_vae(VanillaVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List | None = [512, 256, 128, 64, 32],
                 encoder: Any | None = None,
                 decoder: Any | None = None,
                 dropout_nn: Any | None = None,
                 kl_weight = 0.00025,
                 **kwargs) -> None:

        # Build Encoder
        if encoder is None:
            encoder = FC_block(in_channels + 1, hidden_dims[-1], hidden_dims[:-2])

        # Build Decoder
        if decoder is None:
            hidden_dims.reverse()
            activation_funcs = len(hidden_dims)*[nn.ReLU()] + [nn.Softmax()]
            decoder = FC_block(latent_dim + 1, in_channels, hidden_dims,
                               activations=activation_funcs)
            hidden_dims.reverse()

        super().__init__(in_channels, latent_dim, hidden_dims=hidden_dims,
                         encoder=encoder, decoder=decoder, kl_weight=kl_weight)

        # Build dropout
        if dropout_nn is None:
            dropout_nn = FC_block(self.latent_dim + 1, in_channels, [128],
                                  activations=[nn.ReLU(), nn.Sigmoid()])

        self.dropout_nn = dropout_nn
            
        self.dispersion = torch.rand(in_channels, requires_grad=True, device='cuda')


    def encode(self, input: Tensor, batch: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x F]
        :param batch: (Tensor) batch of cells ([N x 1])
        :return: (Tensor) List of latent codes [[N x D], [N x D]]
        """
        input = torch.cat((input, batch), 1)
        print(input.shape)
        result = self.encoder(input)
        print(result.shape)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]
    
    def decode(self, z: Tensor, batch: Tensor, theta: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :param batch: (Tensor) batch of cells ([N x 1])
        :param theta: (Tensor) dispersion of cells ([N x 1])
        :return: (Tensor) [B x F]
        """
        input = torch.cat((z, batch), 1)
        result = self.decoder(input)
        result = torch.distributions.gamma.Gamma(result, theta).sample()
        return result
    
    def forward(self, input: Tensor, batch, theta, read_depth, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input, batch)
        z = self.reparameterize(mu, log_var)
        proportions = self.decode(z, batch, theta)  #mean gamma (scVI px_scale)

        mean_poisson = proportions * torch.exp(read_depth)  #(scVI px_rate)
        sampled_poisson = torch.distributions.poisson.Poisson(mean_poisson).sample()

        p_drop_out = self.dropout_nn(torch.cat((z, batch), 1))
        sampled_drop_out = torch.distributions.bernoulli.Bernoulli(probs=p_drop_out).sample()
                                                                   
        drop_out_mask = sampled_drop_out.type(torch.bool)^1
        
        observed = drop_out_mask*sampled_poisson
        return  [observed, input, mu, log_var, mean_poisson, p_drop_out]

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
        #recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]
        mean_poisson = args[4]
        p_drop_out = args[4]


        kld_weight =  self.kl_weight
        recons_loss = self.get_zinb_loss(input, mean_poisson, p_drop_out, self.dispersion)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def training_step(self, batch, batch_idx):
        x  =  batch[0]
        batch_id = batch[1]['batch']
        read_depth = batch[1]['read_depth']

        losses = self.loss_function(*self.forward(x, batch_id,
                                                  self.dispersion,
                                                  read_depth))
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        return losses
    
    def test_step(self, batch, batch_idx):
        x  =  batch[0]
        batch_id = batch[1]['batch']
        read_depth = batch[1]['read_depth']

        losses = self.loss_function(*self.forward(x, batch_id,
                                                  self.dispersion,
                                                  read_depth))
        self.log_dict(losses)

    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        x  =  batch[0]
        batch_id = batch[1]['batch']
        read_depth = batch[1]['read_depth']
        losses = self.loss_function(*self.forward(x, batch_id,
                                                  self.dispersion,
                                                  read_depth))
        self.log_dict(losses, on_step=False, on_epoch=True, prog_bar=True)
        val_loss = losses['loss']
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
    
    
    def get_zinb_loss(self, x, mean_poiss, prob_dropout, dispersion):
        where_zero = torch.le(x, EPS)

        zeros = torch.zeros(x.shape)

        case_zero_loss = torch.log(prob_dropout + (1 - prob_dropout)*self.get_nb_zero_loss(mean_poiss, dispersion))

        case_non_zero_loss = self.get_nb_log_loss(x, mean_poiss, dispersion)

        comb_loss = where_zero*case_zero_loss + (~where_zero)*case_non_zero_loss

        return torch.sum(comb_loss)

    def get_nb_log_loss(self, x, mean_poiss, dispersion):
        return torch.lgamma(x + dispersion) - torch.lgamma(x + 1)  - torch.lgamma(dispersion) \
            + x*torch.log(1 - mean_poiss) + dispersion*torch.log(mean_poiss)
    
    def get_nb_zero_loss(self, mean_poiss, dispersion):
        return torch.pow(mean_poiss, dispersion)



