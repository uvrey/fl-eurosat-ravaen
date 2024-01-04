import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import random_split, DataLoader


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
    
    