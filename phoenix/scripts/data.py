import hydra
from omegaconf import OmegaConf, DictConfig

from pytorch_lightning import seed_everything

import flwr as fl

from utils import deepconvert
from flutils.fldataset import prepare_dataset
from flutils.server import get_on_fit_config, get_evaluate_fn
from flutils.client import generate_client_fn

from data.datamodule import ParsedDataModule

def setup_config():
    config = {
        "dataset": "alpha_multiscene_tiny",
        "data_module_name": "alpha_multiscene",
        "log_dir": "/home/josie/l46/l46-project/RaVAEn-master/outputs/results",
        "cache_dir": "/home/josie/l46/l46-project/RaVAEn-master/outputs/cache"
    }

    fl_config = {
        "channels": ['B2','B3','B4','B5','B6','B7', 'B8'],
        "visualisation_channels": [2, 1, 0],
        "batch_size": None,
        "num_clients": None,
        "fraction_fit": 0.0001,
        "fraction_eval": 0.001,
        "min_fit_clients": 20,
        "min_eval_clients": 20,
        "local_epochs": 10,
        "num_rounds": 5,
        "num_clients": 20,
        "config_fit": {
            "lr": 0.001,
            "momentum": 0.9,
            "local_epochs": 10
        },
        "model_cls_args": {
            "hidden_channels": [32, 64, 128],
            "latent_dim": 128,
            "extra_depth_on_scale": 2
            }
    }
    return config, fl_config


def main():
    
    """
    1. Parse config and get experiment output dir
    """
       
    seed_everything(42, workers=True)
    config, fl_config = setup_config()
    
    """
    ## 2. Prepare dataset
    """
    print("\nPreprocessing dataset...")
    
    data_module = ParsedDataModule.load_or_create(config,
                                                  config['cache_dir'])

    input_shape = data_module.sample_shape_train_ds.to_tuple()[0]
    latent_dim = fl_config['model_cls_args']['latent_dim']
    vis_channels = fl_config['visualisation_channels']

    print(f"Extracted latent_dim={latent_dim} from fl_config dict")
    print(f"Extracted vis_channels={vis_channels} from fl_config dict")
    
    training_set = data_module.train_ds.datasets[2] # the 3rd dataset in the sequence has most of the samples
    test_set = data_module.test_ds.datasets[0]
    
    # (12190, 1) -> (10, 32, 32)
    
    print("\nDatamodule created!")
    #print(f"Training set length: {len(training_set)}")
    #print(f"Test set length: {len(test_set)}")
    
    # prepare the dataloaders where the number of partitions corresponds to number of clients
    train_loaders, val_loaders, test_loader = prepare_dataset(datamodule=data_module,
                                                            batch_size=flcfg['training']['batch_size_train'],
                                                            num_partitions=flcfg['num_clients'])
    
    # debug messages
    print("\nFedRaVAEn dataset loaded!")
    print(f"Number of training loaders: {len(train_loaders)}, Number of Validation Loaders: {len(val_loaders)}")
    print(f"Length of each partition's dataset: {len(train_loaders[0].dataset)} (training), {len(val_loaders[0].dataset)} (validation)")
    print(f"Length of test dataset: {len(test_loader.dataset)}")
    
    """
    ## 3. Define FL clients
    """
    
    # print("\nGenerating clients...")
    
    # client_fn = generate_client_fn(train_loaders, val_loaders, input_shape, latent_dim, vis_channels, flcfg['training'])
    
    # """
    # ## 4. Define FL strategy
    # """
    # strategy = fl.server.strategy.FedAvg(fraction_fit=flcfg['fraction_fit'],
    #                                      min_fit_clients=flcfg['min_fit_clients'],
    #                                      fraction_evaluate=flcfg['fraction_eval'],
    #                                      min_evaluate_clients=flcfg['min_eval_clients'],
    #                                      min_available_clients=flcfg['num_clients'],
    #                                      on_fit_config_fn=get_on_fit_config(flcfg['config_fit']),
    #                                      on_evaluate_config_fn=get_evaluate_fn(input_shape=input_shape, testloader=test_loader))
    
    # """
    # ## 5. Start simulation
    # """
    # history = fl.simulation.start_simulation(client_fn=client_fn,
    #                                          num_clients=flcfg['num_clients'],
    #                                          config=fl.server.ServerConfig(num_rounds=flcfg['num_rounds']),
    #                                          strategy=strategy)
    
    """
    ## 6. Save results
    """
    
if __name__ == "__main__":
    main()