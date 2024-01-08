from collections import OrderedDict

from pytorch_lightning import loggers
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

import pytorch_lightning as pl

import torch

#from src.models.custom_model import SimpleVAEModel

import flwr as fl
import hydra

#from src.callbacks.visualisation_callback import VisualisationCallback

### FROM custom_model.py ############
from torch import nn, Tensor
import torch
from torch.nn import functional as F
from typing import List, Any, Dict, Tuple
#####################################


class RaVAEnClient(fl.client.Client):
    def __init__(self,
                 trainloader,
                 valloader,
                 cid,
                 input_shape,
                 latent_dim,
                 visualisation_channels,
                 cfg_train) -> None:
        super().__init__()

        self.cid = cid # client ID
        self.cfg_train = cfg_train # config file from main.py
        
        self.trainloader = trainloader
        self.valloader = valloader
        
        # define the model - SimpleVAE
        self.model = SimpleVAEModel(input_shape=input_shape, latent_dim=latent_dim, visualisation_channels=visualisation_channels)
        
        # use GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO: Check if need to separate encoder and decoder params
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # TODO: Check if need to separate encoder and decoder params
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters):
        
        self.set_parameters(parameters)
        
        #logger = loggers.WandbLogger(save_dir=self.config['log_dir'], name=self.config,
        #                            project=self.config['project'], entity=self.config['entity'])
        
        # callbacks = [
        #     VisualisationCallback(),
        #     LearningRateMonitor(),
        #     ModelCheckpoint(
        #         save_last=True,
        #         save_top_k=-1,  # -1 keeps all, # << 0 keeps only last ....
        #         filename='epoch_{epoch:02d}-step_{step}',
        #         auto_insert_metric_name=False)
        #     ]

        # plugins = []
        # if cfg_train.get('distr_backend') == 'ddp':
        #     plugins.append(DDPPlugin(find_unused_parameters=False))
        
        # TODO: Set these parameters based on the flwr on_fit_config_fn         
        trainer = pl.Trainer(
            deterministic=True,
            gpus=self.cfg_train['gpus'],
            # logger=logger,
            # callbacks=callbacks,
            # plugins=plugins,
            profiler='simple',
            max_epochs=self.cfg_train['epochs'],
            accumulate_grad_batches=self.cfg_train['grad_batches'],
            accelerator=self.cfg_train.get('distr_backend'),
            precision=16 if self.cfg_train['use_amp'] else 32,
            auto_scale_batch_size=self.cfg_train.get('auto_batch_size'),
            auto_lr_find=self.cfg_train.get('auto_lr', False),
            check_val_every_n_epoch=self.cfg_train.get('check_val_every_n_epoch', 10),
            reload_dataloaders_every_epoch=False,
            fast_dev_run=self.cfg_train['fast_dev_run'],
            resume_from_checkpoint=self.cfg_train.get('from_checkpoint')
        )
        
        trainer.fit(self.model, self.train_loader, self.val_loader)

        return self.get_parameters(), len(self.trainloader), {}
        
    def evaluate(self, parameters):
        self.set_parameters(parameters)

        trainer = pl.Trainer()
        results = trainer.test(self.model, self.test_loader)
        loss = results[0]["test_loss"]
        accuracy = results[0]["accuracy"]

        return loss, len(self.testloader), {"accuracy": accuracy}
    

def generate_client_fn(trainloaders, valloaders, input_shape, latent_dim, visualisation_channels, cfg_train):
    def client_fn(cid: str):
        return RaVAEnClient(trainloader=trainloaders[int(cid)], 
                            valloader=valloaders[int(cid)], 
                            input_shape=input_shape,
                            latent_dim=latent_dim,
                            visualisation_channels=visualisation_channels,
                            cfg_train=cfg_train)
    return client_fn

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
    