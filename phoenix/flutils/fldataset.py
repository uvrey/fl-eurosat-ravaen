import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader, Subset

"""
Generates training, validation and test dataloaders for FedRaVAEn.

Splits the training set into equal partitions based on the number of training examples
and the number of clients, and then splits each partition into its own training and
validation set. Creates DataLoader objects for each partition as well as the test set.

Params:
    datamodule: pytorch lightning datamodule class (so we can use the RaVAEn datamodule implementation)
    batch_size: batch size
    num_partitions: number of partitions to partition the training set into. Equal to the number of FL clients
    val_ratio: ratio by which to split partition into training and validation sets
    random_seed: random seed for sampling the dataset when creating partitions
    
Returns:
    trainloaders: List of DataLoader objects for training sets of each partition
    valloaders: List of DataLoader objects for validation sets of each partition
    testloader: DataLoader object for the test set
"""
def prepare_dataset(datamodule: LightningDataModule, 
                    batch_size: int,
                    num_partitions: int, 
                    val_ratio: float = 0.1,
                    random_seed: int = 2024):
    
    # TODO Change this to extract the training set from RaVAEn dataset    
    train_set = datamodule.train_ds.datasets[2] # the 3rd dataset in the sequence has most of the samples
    test_set = datamodule.test_ds.datasets[0]
    
    len_train_original = len(train_set)
    
    # number of images per partition
    num_images = len(train_set) // num_partitions
    
    # lengths of each equal partition (constant)
    partition_lengths = [num_images] * (num_partitions)
    
    # remove the left-over entries which are insufficient to form another partition
    train_set = Subset(train_set, range(num_images*num_partitions))

    # append the remaining images to the list of partition lengths
    #images_left_over = len(train_set) - num_images * num_partitions
    #partition_lengths.append(images_left_over) 
    
    print(f"Partitioned {len(train_set)} of {len_train_original} training set entries into {num_partitions} partitions of length {partition_lengths[0]}")
    
    print(f"Training set length: {len(train_set)}")
    print(f"Sum of partitions: {sum(partition_lengths)}")
    
    # randomly partition the training set 
    # with equal partition length of partition_len
    train_sets = random_split(train_set, 
                              partition_lengths, 
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
    
    