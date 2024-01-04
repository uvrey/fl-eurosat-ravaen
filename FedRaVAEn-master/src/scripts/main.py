
import hydra
from omegaconf import OmegaConf, DictConfig

from pytorch_lightning import seed_everything

import flwr as fl

from .fldataset import prepare_dataset
from .utils import deepconvert
from .server import get_on_fit_config, get_evaluate_fn
from .client import generate_client_fn

from src.data.datamodule import ParsedDataModule

@hydra.main(config_path="../config", config_name="flconfig")
def main(flcfg: DictConfig):
    
    """
    1. Parse config and get experiment output dir
    """
    print("Contents of flconfig.yaml:")
    print(OmegaConf.to_yaml(flcfg))
       
    seed_everything(42, workers=True)
    flcfg = deepconvert(flcfg)
    
    """
    ## 2. Prepare dataset
    """
    print("Generating dataset...")
    
    data_module = ParsedDataModule.load_or_create(flcfg['dataset'],
                                                  flcfg['cache_dir'])

    input_shape = data_module.sample_shape_train_ds.to_tuple()[0]


    # prepare the dataloaders where the number of partitions corresponds to number of clients
    train_loaders, val_loaders, test_loader = prepare_dataset(datamodule=data_module,
                                                            batch_size=flcfg['training']['batch_size_train'],
                                                            num_partitions=flcfg['num_clients'])
    
    # debug messages
    num_trainloaders = len(train_loaders)
    len_trainloader = len(train_loaders[0].dataset)
    print("RaVAEn dataset loaded!")
    print(f"Number of training loaders: {num_trainloaders}")
    print(f"Trainloader Length: {len_trainloader}")
    
    """
    ## 3. Define FL clients
    """
    
    client_fn = generate_client_fn(train_loaders, val_loaders, input_shape, flcfg)
    
    """
    ## 4. Define FL strategy
    """
    strategy = fl.server.strategy.FedAvg(fraction_fit=flcfg['fraction_fit'],
                                         min_fit_clients=flcfg['min_fit_clients'],
                                         fraction_evaluate=flcfg['fraction_eval'],
                                         min_evaluate_clients=flcfg['min_eval_clients'],
                                         min_available_clients=flcfg['num_clients'],
                                         on_fit_config_fn=get_on_fit_config(flcfg['config_fit']),
                                         on_evaluate_config_fn=get_evaluate_fn(input_shape=input_shape, testloader=test_loader)
                                         )
    
    """
    ## 5. Start simulation
    """
    
    
    """
    ## 6. Save results
    """
    
if __name__ == "__main__":
    main()