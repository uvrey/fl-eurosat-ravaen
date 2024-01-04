from collections import OrderedDict

from pytorch_lightning import loggers
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin

import pytorch_lightning as pl

import torch

from src.models.ae_vae_models.simple_vae import SimpleVAE

import flwr as fl
import hydra

from visualisation_callback import VisualisationCallback

class RaVAEnClient(fl.client.Client):
    def __init__(self,
                 trainloader,
                 valloader,
                 cid,
                 input_shape,
                 config) -> None:
        super().__init__()

        self.cid = cid # client ID
        self.config = config # config file from main.py
        
        self.trainloader = trainloader
        self.valloader = valloader
        
        # define the model - SimpleVAE
        self.model = SimpleVAE(input_shape=input_shape, **config['module']['model_cls_args'])
        
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
            
        cfg_train = self.config['training']
        
        logger = loggers.WandbLogger(save_dir=self.config['log_dir'], name=self.config,
                                    project=self.config['project'], entity=self.config['entity'])
        
        callbacks = [
            VisualisationCallback(),
            LearningRateMonitor(),
            ModelCheckpoint(
                save_last=True,
                save_top_k=-1,  # -1 keeps all, # << 0 keeps only last ....
                filename='epoch_{epoch:02d}-step_{step}',
                auto_insert_metric_name=False)
            ]

        plugins = []
        if cfg_train.get('distr_backend') == 'ddp':
            plugins.append(DDPPlugin(find_unused_parameters=False))
        
        trainer = pl.Trainer(
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
        
    def evaluate(self, parameters):
        self.set_parameters(parameters)

        trainer = pl.Trainer()
        results = trainer.test(self.model, self.test_loader)
        loss = results[0]["test_loss"]
        accuracy = results[0]["accuracy"]

        return loss, len(self.testloader), {"accuracy": accuracy}
    

def generate_client_fn(trainloaders, valloaders, input_shape, cfg):
    def client_fn(cid: str):
        return RaVAEnClient(trainloader=trainloaders[int(cid)], 
                            valloader=valloaders[int(cid)], 
                            input_shape=input_shape, 
                            config=cfg)
    return client_fn