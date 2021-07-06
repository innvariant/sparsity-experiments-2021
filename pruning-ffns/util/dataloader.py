import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_valid_dataset(train_batch=64, valid_batch=None):
    """
    Creates a trainings and cross-validation dataset out of the original train dataset. The validation set contains
    10,000 images and the test set contains 50,000 images.

    :param train_batch:         The size of each training set batch.
    :param valid_batch:         The size of each validation set batch.
    :return: train_dataset: The dataset that is used for training of the network.
    :return: valid_dataset: The dataset that is used for the cross validation in the network.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset = datasets.MNIST('./dataset', train=True, download=True, transform=transform)
    # valid_dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = 10000

    # shuffle dataset
    np.random.seed(0)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    if valid_batch is None:
        btx = split
    else:
        btx = valid_batch

    # create loader for train and validation sets
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=train_batch, sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=btx, sampler=valid_sampler)

    return train_loader, validation_loader


def get_test_dataset():
    """
    Get the test dataset loaded

    :return: The test dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    return torch.utils.data.DataLoader(
        datasets.MNIST('./dataset', train=False, download=True, transform=transform),
        batch_size=100, shuffle=True)
