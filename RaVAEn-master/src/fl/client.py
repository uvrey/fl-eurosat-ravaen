import argparse
from collections import OrderedDict

from pytorch_lightning import Trainer, loggers, seed_everything
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

import pytorch_lightning as pl

import torch
from datasets.utils.logging import disable_progress_bar

import flwr as fl
import hydra

#disable_progress_bar()

class RaVAEnClient(fl.client.Client):
    def __init__(self,
                 model,
                 trainloader,
                 valloader,
                 cid) -> None:
        super().__init__()

        self.cid = cid
        
        self.trainloader = trainloader
        self.valloader = valloader
        
        self.model = model
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    def get_parameters(self, config):
        encoder_params = _get_parameters(self.model.encoder)
        decoder_params = _get_parameters(self.model.decoder)
        return encoder_params + decoder_params

    def set_parameters(self, parameters):
        _set_parameters(self.model.encoder, parameters[:4]) # encoder params?
        _set_parameters(self.model.decoder, parameters[4:]) # decoder params?
    
    def fit(self, parameters, config):
        
        self.set_parameters(parameters)

        trainer = pl.Trainer(
            max_epochs=5,
            deterministic=True,
            gpus=cfg_train['gpus'],
            logger=logger,
            callbacks=callbacks,
            plugins=plugins,
            profiler='simple',
            max_epochs=cfg_train['epochs'],
            accumulate_grad_batches=cfg_train['grad_batches'],
            accelerator=cfg_train.get('distr_backend'),
            precision=16 if cfg_train['use_amp'] else 32,
            auto_scale_batch_size=cfg_train.get('auto_batch_size'),
            auto_lr_find=cfg_train.get('auto_lr', False),
            check_val_every_n_epoch=cfg_train.get('check_val_every_n_epoch', 10),
            reload_dataloaders_every_epoch=False,
            fast_dev_run=cfg_train['fast_dev_run'],
            resume_from_checkpoint=cfg_train.get('from_checkpoint')
        )
        
        trainer.fit(self.model, self.train_loader, self.val_loader)

        return self.get_parameters(config={}), len(self.trainloader), {}
        
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer()
        results = trainer.test(self.model, self.test_loader)
        loss = results[0]["test_loss"]
        accuracy = results[0]["accuracy"]

        return loss, len(self.testloader), {"loss": loss, "accuracy": accuracy}


def _get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def main() -> None:

    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--node-id",
        type=int,
        choices=range(0, 10),
        required=True,
        help="Specifies the artificial data partition",
    )
    args = parser.parse_args()
    node_id = args.node_id

    # Model and data
    model = mnist.LitAutoEncoder()
    train_loader, val_loader, test_loader = mnist.load_data(node_id)

    # Flower client
    client = FlowerClient(model, train_loader, val_loader, test_loader)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

if __name__ == "__main__":
    main()