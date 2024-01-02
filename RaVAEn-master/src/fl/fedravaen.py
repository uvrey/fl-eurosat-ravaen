import torch

#from torch import nn
#from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader
#from torchvision import transforms

# from flwr_datasets import FederatedDataset

from pytorch_lightning import LightningDataModule, seed_everything
import pytorch_lightning as pl

import hydra

from src.utils import load_obj, deepconvert
from src.data.datamodule import ParsedDataModule
from src.callbacks.visualisation_callback import VisualisationCallback
from src.models.ae_vae_models import simple_vae

def prepare_dataset(datamodule: LightningDataModule, 
                    batch_size: int,
                    num_partitions: int, 
                    val_ratio: float = 0.1,
                    random_seed: int = 2024):
    
    train_set = datamodule.train_ds
    #val_set = datamodule.val_ds
    test_set = datamodule.test_ds
    
    # number of images per partition
    num_images = datamodule.len_train_ds // num_partitions
    
    # length of each partition (constant)
    partition_len = [num_images] * num_partitions
    
    # randomly partition the training set 
    # with equal partition length of partition_len
    train_sets = random_split(train_set, 
                              partition_len, 
                              torch.Generator().manual_seed(random_seed))
    
    # create dataloaders (each partition with train+val support)
    trainloaders = []
    valloaders = []
    
    for train_set_ in train_sets:
        num_total = len(train_set_) # current partition length
        num_val = int(val_ratio * num_total) # number of val examples
        num_train = num_total - num_val # number of train examples
        
        for_train, for_val = random_split(train_set_,
                                          [num_train, num_val],
                                          torch.Generator().manual_seed(random_seed))
        
        trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
        valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))
        
    testloader = DataLoader(test_set, batch_size=128)
    
    return trainloaders, valloaders, testloader

@hydra.main(config_path='../config', config_name='config.yaml')
def main(cfg):
    """Centralized training."""

    # hydra load and create config     
    seed_everything(42, workers=True)

    cfg = deepconvert(cfg)

    data_module = ParsedDataModule.load_or_create(cfg['dataset'],
                                                  cfg['cache_dir'])

    cfg['module']['len_train_ds'] = data_module.len_train_ds
    cfg['module']['len_val_ds'] = data_module.len_val_ds
    cfg['module']['len_test_ds'] = data_module.len_test_ds

    cfg['module']['input_shape'] = data_module.sample_shape_train_ds.to_tuple()[0]

    cfg_train = cfg['training']
    module = load_obj(cfg['module']['class'])(cfg['module'], cfg_train)

    log_name = cfg['module']['class'] + '/' + cfg['project']
    #logger = loggers.WandbLogger(save_dir=cfg['log_dir'], name=log_name,
    #                             project=cfg['project'], entity=cfg['entity'])
    
    callbacks = [
        VisualisationCallback(),
        LearningRateMonitor(),
        ModelCheckpoint(
            save_last=True,
            save_top_k=-1,  # -1 keeps all, # << 0 keeps only last ....
            filename='epoch_{epoch:02d}-step_{step}',
            auto_insert_metric_name=False)
    ]
    
    # Load RaVAEn data
    train_loader, val_loader, test_loader = prepare_dataset(datamodule=data_module,
                                                            batch_size=32,
                                                            num_partitions=10)

    # Load model (using SimpleVAE for now)
    model = simple_vae.SimpleVAE()

    # Train
    trainer = pl.Trainer(   max_epochs=5,
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
                            resume_from_checkpoint=cfg_train.get('from_checkpoint'),    
                            )
    
    trainer.fit(model, train_loader, val_loader)

    # Test
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()