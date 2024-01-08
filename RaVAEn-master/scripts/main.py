
import hydra
from omegaconf import OmegaConf, DictConfig
import src as src
from pytorch_lightning import seed_everything

import flwr as fl
import ray
import yaml

from src.utils import deepconvert
from src.flutils.fldataset import prepare_dataset
from src.flutils.server import get_on_fit_config, get_evaluate_fn
from src.flutils.client import generate_client_fn

from src.data.datamodule import ParsedDataModule
from src.flutils.client import SimpleVAEModel, get_parameters

@hydra.main(config_path="../config", config_name="config")
def main(flcfg: DictConfig):
    
    """
    1. Parse config and get experiment output dir
    """
    #print("Contents of flconfig.yaml:")
    #print(OmegaConf.to_yaml(flcfg))
       
    seed_everything(42, workers=True)
    flcfg = deepconvert(flcfg)
    
    """
    ## 2. Prepare dataset
    """
    print("\nPreprocessing dataset...")
    
    data_module = ParsedDataModule.load_or_create(flcfg['dataset'],
                                                  flcfg['cache_dir'])

    input_shape = data_module.sample_shape_train_ds.to_tuple()[0]
    latent_dim = flcfg['module']['model_cls_args']['latent_dim']
    vis_channels = flcfg['channels']['visualisation_channels']

    print(f"Extracted latent_dim={latent_dim} from config.yaml")
    print(f"Extracted vis_channels={vis_channels} from config.yaml")
    
    training_set = data_module.train_ds.datasets[2] # the 3rd dataset in the sequence has most of the samples
    test_set = data_module.test_ds.datasets[0]
    
    # (12190, 1) -> (10, 32, 32)
    print("\nDatamodule created!")
    
    # prepare the dataloaders where the number of partitions corresponds to number of clients
    train_loaders, val_loaders, test_loader = prepare_dataset(datamodule=data_module,
                                                            batch_size=flcfg['training']['batch_size_train'],
                                                            num_partitions=flcfg['num_clients'])
    
    # debug messages
    print("\nFedRaVAEn dataset loaded!")
    print(f"Number of training loaders: {len(train_loaders)}, Number of Validation Loaders: {len(val_loaders)}")
    print(f"Length of each partition's dataset: {len(train_loaders[0].dataset)} (training), {len(val_loaders[0].dataset)} (validation)")
    print(f"Length of test dataset: {len(test_loader.dataset)}")
    
    # Load Ray environment
    # Load the contents of the env.yaml file
    with open('/home/josie/l46/l46-project/RaVAEn-master/env.yaml', 'r') as file:
        env_config = yaml.safe_load(file)

    # Initialize Ray with the runtime environment from the YAML file
    print("Initialising Ray runtime..")
   # ray.init(runtime_env=env_config)
    ray.init(runtime_env={"working_dir": "/home/josie/l46/l46-project/RaVAEn-master/"}, object_store_memory=10**9)
    print("Success!")

    """
    ## 3. Define FL clients
    """
    print("\nGenerating clients...")
    
    client_fn = generate_client_fn(train_loaders, val_loaders, input_shape, latent_dim, vis_channels, flcfg['training'])
    print("Success!")
    
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


    ### TEST
    # Create an instance of the model and get the parameters
    params = get_parameters(SimpleVAEModel(input_shape, latent_dim, vis_channels))

    # Pass parameters to the Strategy for server-side parameter initialization
    print("Creating strategy...")

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.3,
        fraction_evaluate=0.3,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=flcfg['num_clients'],
        initial_parameters=fl.common.ndarrays_to_parameters(params),
    )
    print("Success!")

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    DEVICE = "cpu" # TODO change
    # if DEVICE.type == "cuda":
    #     client_resources = {"num_gpus": 1}

    # Start simulation
    runtime_env = {"py_modules": [src]}
    ignore_reinit_error = True
    namespace="RaVAEn-master"

    print("Starting Simulation...")
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=flcfg['num_clients'],
        config=fl.server.ServerConfig(num_rounds=3),  # Just three rounds
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args =  {'num_cpus': 8, 'num_gpus': 1, "runtime_env": runtime_env}
    )
    print("Success!")
    # """
    # ## 5. Start simulation
    # """
    # runtime_env = {"py_modules": [src]}

    # history = fl.simulation.start_simulation(client_fn=client_fn,
    #                                          num_clients=flcfg['num_clients'],
    #                                          config=fl.server.ServerConfig(num_rounds=flcfg['num_rounds']),
    #                                          strategy=strategy,
    #                                          ray_init_args =  {'num_cpus': 8, 'num_gpus': 1, "runtime_env": runtime_env}
    #                                          )
    """
    ## 6. Save results
    """
    
if __name__ == "__main__":
    main()