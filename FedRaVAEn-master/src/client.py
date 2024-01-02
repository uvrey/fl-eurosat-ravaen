from typing import Optional
from pathlib import Path

import flwr as fl
import torch
from torch.utils.data import DataLoader
from flwr.common import EvaluateIns, EvaluateRes, FitIns, FitRes, GetPropertiesIns, GetPropertiesRes, GetParametersIns, \
                        GetParametersRes, Status, Code, parameters_to_ndarrays, ndarrays_to_parameters

from network import Net, train, test


from dataset_utils import get_dataset


def get_dataloader(path_to_data: str, cid: str, partition: str, batch_size: int):
    """Generates trainset/valset object and returns appropiate dataloader."""
    dataset = get_dataset(Path(path_to_data), cid, partition)
    return DataLoader(dataset, batch_size=batch_size, pin_memory=True,
                      shuffle=(partition == "train"))


class RaVAEnClient(fl.client.Client):
    def __init__(self, cid: str, fed_data_dir: str, log_progress: bool = False):
        """
        Creates a client for training `simpleVAE` on World Floods Data.

        Args:
            cid: A unique ID given to the client (typically a number)
            fed_data_dir: A path to a partitioned dataset
            log_progress: Controls whether clients log their progress
        """
        self.cid = cid
        self.data_dir = fed_data_dir
        self.properties = {"tensor_type": "torch.Tensor"}
        self.log_progress = log_progress
        
        # Initilise the `net`` variable to `None`
        self.net = None
        
        # determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    def get_properties(self):
        return GetPropertiesRes(properties=self.properties)
    
    def get_parameters(self):
        if self.net is None:
            self.net = Net()
        return GetParametersRes(status=Status(Code.OK, ""), parameters=ndarrays_to_parameters(self.net.get_weights()))

class CifarClient(fl.client.Client):
    def __init__(self, cid: str, fed_data_dir: str, log_progress: bool = False):
        """
        Creates a client for training `network.Net` on CIFAR-10.

        Args:
            cid: A unique ID given to the client (typically a number)
            fed_data_dir: A path to a partitioned dataset
            log_progress: Controls whether clients log their progress
        """
        self.cid = cid
        self.data_dir = fed_data_dir
        self.properties = {"tensor_type": "numpy.ndarray"}
        self.log_progress = log_progress
        # Initilise the `net`` variable to `None`
        self.net = None

        # determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_properties(self, ins: GetPropertiesIns):
        return GetPropertiesRes(properties=self.properties)

    def get_parameters(self, ins: GetParametersIns):
        if self.net is None:
            self.net = Net()
        return GetParametersRes(status=Status(Code.OK, ""), parameters=ndarrays_to_parameters(self.net.get_weights()))

    def set_parameters(self, parameters):
        if self.net is None:
            self.net = Net()
        self.net.set_weights(parameters_to_ndarrays(parameters))

    def fit(self, fit_params: FitIns) -> FitRes:
        # Instantiate model (best practise)
        self.net = Net()
        # Process incoming request to train
        batch_size = fit_params.config["batch_size"]
        num_iterations = fit_params.config["num_iterations"]
        self.set_parameters(fit_params.parameters)

        # Initialise data loader
        trainloader = get_dataloader(
            path_to_data=self.data_dir,
            cid=self.cid,
            partition="train",
            batch_size=batch_size)

        # num_iterations = None special behaviour: train(...) runs for a single epoch, however many updates it may be
        num_iterations = num_iterations or len(trainloader)

        # Train the model
        print(f"Client {self.cid}: training for {num_iterations} iterations/updates")
        self.net.to(self.device)
        train_loss, train_acc, num_examples = \
            train(self.net, trainloader, device=self.device,
                  num_iterations=num_iterations, log_progress=self.log_progress)
        print(f"Client {self.cid}: training round complete, {num_examples} examples processed")

        # Return training information: model, number of examples processed and metrics
        return FitRes(
            status=Status(Code.OK, ""),
            parameters=self.get_parameters(fit_params.config).parameters,
            num_examples=num_examples,
            metrics={"loss": train_loss, "accuracy": train_acc})

    def evaluate(self, eval_params: EvaluateIns) -> EvaluateRes:
        # Process incoming request to evaluate
        batch_size = eval_params.config["batch_size"]
        self.set_parameters(eval_params.parameters)

        # Initialise data loader
        valloader = get_dataloader(
            path_to_data=self.data_dir,
            cid=self.cid,
            partition="val",
            batch_size=batch_size)

        # Evaluate the model
        self.net.to(self.device)
        loss, accuracy, num_examples = test(self.net, valloader, device=self.device, log_progress=self.log_progress)

        print(f"Client {self.cid}: evaluation on {num_examples} examples: loss={loss:.4f}, accuracy={accuracy:.4f}")
        # Return evaluation information
        return EvaluateRes(
            status=Status(Code.OK, ""),
            loss=loss, num_examples=num_examples,
            metrics={"accuracy": accuracy})