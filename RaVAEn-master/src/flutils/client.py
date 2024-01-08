from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from typing import List, Any, Dict, Tuple
import importlib
import omegaconf

import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from collections import OrderedDict

from pytorch_lightning import loggers, LightningModule
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
import numpy as np


DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

def load_obj(obj_path):
    """
    Call an object from a string
    """

    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0)
    
    print(f"Object path: {obj_path}")
    
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(
            f"Object '{obj_name}' cannot be loaded from '{obj_path}'."
        )
    return getattr(module_obj, obj_name)

class RaVAENClient(fl.client.NumPyClient):
    def __init__(self, cid, model, trainloader, valloader, cfg_train):
        self.cid = cid
        self.net = model
        self.trainloader = trainloader
        self.valloader = valloader
        self.cfg_train = cfg_train

    def get_parameters(self, config):
        print(f"[Client {self.cid}] get_parameters")
        return get_parameters(self.net)

    def fit(self, parameters, config):
        print(f"[Client {self.cid}] fit, config: {config}")
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=1)
        return get_parameters(self.net), len(self.trainloader), {}
        # set_parameters(self.net, parameters)
        
        # # TODO: FIX. Option 1 - use a trainer. Otherwise, define training function   
        # trainer = pl.Trainer(
        #     deterministic=True,
        #     # gpus=self.cfg_train['gpus'], # TODO would need to check if available
        #     profiler='simple',
        #     max_epochs=self.cfg_train['epochs'],
        #     accumulate_grad_batches=self.cfg_train['grad_batches'],
        #     accelerator=self.cfg_train.get('distr_backend'),
        #     # precision=16 if self.cfg_train['use_amp'] else 32, # TODO add back as this is GPU specific 
        #     auto_scale_batch_size=self.cfg_train.get('auto_batch_size'),
        #     auto_lr_find=self.cfg_train.get('auto_lr', False),
        #     check_val_every_n_epoch=self.cfg_train.get('check_val_every_n_epoch', 10),
        #     reload_dataloaders_every_epoch=False,
        #     fast_dev_run=self.cfg_train['fast_dev_run'],
        #     resume_from_checkpoint=self.cfg_train.get('from_checkpoint')
        # )
        
        # trainer.fit(self.net, self.trainloader, self.valloader)
        # return get_parameters(self.net), len(self.trainloader), {}
        # return self.get_parameters(self.cfg_train), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        print(f"[Client {self.cid}] evaluate, config: {config}")
        set_parameters(self.net, parameters)
        loss = test(self.net, self.valloader)
        return float(loss), len(self.valloader)

def generate_client_fn(trainloaders, valloaders, input_shape, latent_dim, vis_channels, cfg_train):
    def client_fn(cid: str) -> RaVAENClient:
        model = SimpleVAEModel(input_shape, latent_dim, vis_channels).to(DEVICE)
        return RaVAENClient(cid = cid,
                    model = model,
                    trainloader=trainloaders[int(cid)], 
                    valloader=valloaders[int(cid)], 
                    cfg_train = cfg_train
                    )
    return client_fn

# TODO change back to nn.Module potentially 
class SimpleVAEModel(nn.Module):
    def __init__(self,
                 input_shape: Tuple[int],
                 latent_dim: int,
                 visualisation_channels,
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.visualisation_channels = visualisation_channels # maybe specific to ravaen dataset TODO
        self.input_shape = input_shape

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

    # TODO Problem - inputs are list instead of Tensors
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


## FUNCTIONS 
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

def train(net, trainloader, epochs: int):
    """Train the network on the training set."""
    # TODO replace this with an appropriate training function (ie. FROM RAEVEN!)
    criterion = nn.MSELoss()  # Use Mean Squared Error as the reconstruction loss
    optimizer = optim.Adam(net.parameters())
    net.train()

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in tqdm(trainloader):
            images = batch[0] # from observation
            optimizer.zero_grad()
            reconstructed_images = net(images)[0] # returns List[reconstr_img, mu, var]
            loss = criterion(reconstructed_images, images)  # Reconstruction loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(trainloader.dataset)
        print(f"Epoch {epoch + 1}: train loss {epoch_loss}")

def test(net, testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    loss = 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images = batch[0]
            outputs = net(images)[0]
            loss += criterion(outputs).item()
            _, predicted = torch.max(outputs.data, 1)
    loss /= len(testloader.dataset)
    return loss