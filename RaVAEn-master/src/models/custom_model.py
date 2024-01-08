
from torch import nn, Tensor
import torch
from torch.nn import functional as F
from typing import List, Any, Dict, Tuple

"""
Summarisation of RaVAEn model outlines
"""
class SimpleVAEModel(nn.Module):

    def __init__(self,
                 input_shape: Tuple[int],
                 latent_dim: int,
                 visualisation_channels,
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.visualisation_channels = visualisation_channels # maybe specific to ravaen dataset TODO
        
        # Reconstructing things, so in and out channels should be the same
        in_channels = input_shape[0]
        out_channels = input_shape[0]

        # Defines encoder 
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 7),
            nn.LeakyReLU()
        )

        self.width = (input_shape[1] // 2) // 2 - 6

        # Encodes latent space, it's Linear. Latent dim = output features
        self.fc_mu = nn.Linear(256 * self.width * self.width, latent_dim)
        self.fc_var = nn.Linear(256 * self.width * self.width, latent_dim)

        self.decoder_input = \
            nn.Linear(latent_dim, 256 * self.width * self.width)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 7),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1,
                               output_padding=1),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, out_channels, 3, stride=2, padding=1,
                               output_padding=1),
        )

        # TODO: Check if need to separate encoder and decoder params
    def get_parameters(self, config):
        print(" GETTING THE PARAMETERS!")
        param = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        print(param[0])
        print("------------------------------")
        return param

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
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
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(-1, 256, self.width, self.width)
        result = self.decoder(result)
        return result

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(torch.nan_to_num(input))
        z = self.reparameterize(mu, log_var)
        return [self.decode(z), mu, log_var]

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """
        return self.forward(x)[0]

    def loss_function(self, input: Tensor, results: Any, **kwargs) -> Dict:
        """
        Computes the VAE loss function.

        :param args:
        :param kwargs:
        :return:
        """
        # invalid_mask = torch.isnan(input)
        input = torch.nan_to_num(input)

        recons = results[0]
        mu = results[1]
        log_var = results[2]

        # Account for the minibatch samples from the dataset
        kld_weight = kwargs['M_N']

        recons_loss = F.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        # KLD weight is a hyperparameter, affects how much KLD loss govern training TODO experiment?
        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss,
                'Reconstruction_Loss': recons_loss,
                'KLD': -kld_loss}
        return {'loss': recons_loss, 'Reconstruction_Loss': recons_loss}

    def _visualise_step(self, batch):
        result = self.forward(batch) # [reconstruction, mu, log_var]
        # Just select the reconstruction
        result = result[0]
        rec_error = (batch - result).abs()

        return batch[:, self.visualisation_channels], \
            result[:, self.visualisation_channels], \
            rec_error.max(1)[0]
            
    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1) - Normalises this
        :param mu: Mean of the latent Gaussian [B x D]
        :param logvar: Standard deviation of the latent Gaussian [B x D]
        :return: [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self,
               num_samples: int,
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
            
    @property
    def _visualisation_labels(self):
        return ["Input", "Reconstruction", "Rec error"]
    

## Flower functions
def get_parameters(model) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)