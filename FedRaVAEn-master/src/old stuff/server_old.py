import argparse
import functools
import torch
import click

import flwr as fl

from torch.utils.data import DataLoader
from flwr.server.strategy import FedAvg
from flwr.server.app import ServerConfig
from flwr.common import NDArrays

from dataset_utils import get_cifar_10, do_fl_partitioning
from network import Net, test
from client import CifarClient, RaVAEnClient


def serverside_eval(server_round, parameters: NDArrays, config, testloader):
    """An evaluation function for centralized/serverside evaluation over the entire CIFAR-10 test set."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Net()
    model.set_weights(parameters)
    model.to(device)
    loss, accuracy, _ = test(model, testloader, device=device, log_progress=False)
    print(f"Evaluation on the server: test_loss={loss:.4f}, test_accuracy={accuracy:.4f}")
    return loss, {"accuracy": accuracy}


@click.command()
@click.option("--num-rounds", default=10, help="The number of rounds in the FL experiment")
@click.option("--client-pool-size", default=5, help="The number of clients made available to an FL round")
@click.option("--num-iterations", default=None, type=int,
              help="Number of iterations/updates a client performs per round (single local epoch if None)")
@click.option("--fraction-fit", default=1.0,
              help="Controls what fraction of clients should be sampled for an FL fitting round.")
@click.option("--min-fit-clients", default=2,
              help="The minimum number of clients to participate in a fitting round (regardless of 'fraction_fit')")
@click.option("--batch-size", default=32, help="Batch size for a client fitting round")
@click.option("--val-ratio", default=0.1, help="Proportion of local data to reserve as a local test set")
@click.option("--iid-alpha", default=1000.0, help="LDA prior concentration parameter for data partitioning")
def start_experiment(
    num_rounds=10,
    client_pool_size=5,
    num_iterations=None,
    fraction_fit=1.0,
    min_fit_clients=2,
    batch_size=32,
    val_ratio=0.1,
    iid_alpha=1000.0):
    client_resources = {"num_cpus": 0.5}  # 2 clients per CPU

    # Download the dataset
    #train_path, testset = get_cifar_10()

    # Partition the dataset into subsets reserved for each client.
    # - to control the degree of IID: use a large `alpha` to make it IID; a small value (e.g. 1) will make it non-IID
    # - 'val_ratio' controls the proportion of the (local) client reserved as a local test set
    # (good for testing how the final model performs on the client's local unseen data)
    fed_data_dir = do_fl_partitioning(train_path, pool_size=client_pool_size,
                                      alpha=iid_alpha, num_classes=10, val_ratio=val_ratio)
    
    print(f"Data partitioned across {client_pool_size} clients, with IID alpha = {iid_alpha} "
          f"and {val_ratio} of local dataset reserved for validation.")

    testloader = DataLoader(dataset=testset, batch_size=batch_size, shuffle=False)

    # Configure the strategy
    def fit_config(server_round: int):
        print(f"Configuring round {server_round}")
        return {
            "num_iterations": num_iterations,
            "batch_size": batch_size,
        }

    # FedAvg simply averages contributions from all clients
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit if val_ratio > 0.0 else 0.0,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_fit_clients,
        min_available_clients=client_pool_size,  # all clients should be available
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=(lambda r: {"batch_size": 100}),
        evaluate_fn=functools.partial(serverside_eval, testloader=testloader),
        accept_failures=False,
    )

    print(f"FL experiment configured for {num_rounds} rounds with {client_pool_size} client in the pool.")
    print(f"FL round will proceed with {fraction_fit * 100}% of clients sampled, at least {min_fit_clients}.")

    """
    Generate a RaVAEn client
    """
    #def generate_client_fn(trainloaders, valloaders):
    def client_fn(cid: str):
        return RaVAEnClient(trainloader=trainloaders[int(cid)], valloader=valloaders[int(cid)])
    #return client_fn

    
    #def client_fn(cid: str):
    #    """Creates a federated learning client"""
    #    return CifarClient(cid, fed_data_dir, log_progress=False)

    # Start the simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=client_pool_size,
        client_resources=client_resources,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy)

    print(history)

    return history

if __name__ == "__main__":
    start_experiment()