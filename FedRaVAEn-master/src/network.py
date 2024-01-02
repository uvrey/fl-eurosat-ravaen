from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Optional

import flwr as fl


class Net(nn.Module):
    """Simple CNN adapted from 'PyTorch: A 60 Minute Blitz'."""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_weights(self) -> fl.common.NDArrays:
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.state_dict().items()]

    def set_weights(self, weights: fl.common.NDArrays) -> None:
        """Set model weight s from a list of NumPy ndarrays."""
        state_dict = OrderedDict(
            {k: torch.Tensor(v) for k, v in zip(self.state_dict().keys(), weights)}
        )
        self.load_state_dict(state_dict, strict=True)

def train(
    net: Net,
    trainloader: DataLoader,
    device: torch.device,
    num_iterations: int,
    log_progress: bool = True):
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    def cycle(iterable):
        """Repeats the contents of the train loader, in case it gets exhausted in 'num_iterations'."""
        while True:
            for x in iterable:
                yield x

    # Train the network
    net.train()
    total_loss, total_correct, n_samples = 0.0, 0.0, 0
    pbar = tqdm(iter(cycle(trainloader)), total=num_iterations, desc=f'TRAIN') if log_progress else iter(cycle(trainloader))

    # Unusually, this training is formulated in terms of number of updates/iterations/batches processed
    # by the network. This will be helpful later on, when partitioning the data across clients: resulting
    # in differences between dataset sizes and hence inconsistent numbers of updates per 'epoch'.
    for i, data in zip(range(num_iterations), pbar):
        images, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Collected training loss and accuracy statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        total_correct += (predicted == labels).sum().item()

        if log_progress:
            pbar.set_postfix({
                "train_loss": total_loss/n_samples,
                "train_acc": total_correct/n_samples
            })
    if log_progress:
        print("\n")

    return total_loss/n_samples, total_correct/n_samples, n_samples

def test(
    net: Net,
    testloader: DataLoader,
    device: torch.device,
    log_progress: bool = True):
    """Evaluates the network on test data."""
    criterion = nn.CrossEntropyLoss()
    total_loss, total_correct, n_samples = 0.0, 0.0, 0
    net.eval()
    with torch.no_grad():
        pbar = tqdm(testloader, desc="TEST") if log_progress else testloader
        for data in pbar:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)

            # Collected testing loss and accuracy statistics
            total_loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            n_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
    if log_progress:
        print("\n")

    return total_loss/n_samples, total_correct/n_samples, n_samples
