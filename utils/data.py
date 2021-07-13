import os
import time
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_mnist(data_path, batch_size, num_workers, val_ratio, random_seed):
    # Define train/test transforms
    train_transform = transforms.Compose([
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Get MNIST dataset
    train_data = datasets.MNIST(
        root=data_path,
        train=True,
        download=True,
        transform=train_transform)
    test_data = datasets.MNIST(
        root=data_path,
        train=False,
        download=True,
        transform=test_transform)

    # Extract validation examples
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    split_idx = int(np.floor((1-val_ratio)*num_train))
    train_idxs, val_idxs = indices[:split_idx], indices[split_idx:]
    train_sampler = SubsetRandomSampler(train_idxs)
    val_sampler = SubsetRandomSampler(val_idxs)

    # Construct train/val/test dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        sampler=train_sampler,
        num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        sampler=val_sampler,
        num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers)
    if val_ratio == 0:
        val_loader = test_loader
    return train_loader, val_loader, test_loader


def get_cifar(dataset, data_path, batch_size, num_workers, val_ratio, random_seed):
    # Define train/test transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4, padding_mode='edge'),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Get CIFAR10 dataset
    dataset_func = {
        'cifar10': datasets.CIFAR10,
        'cifar100': datasets.CIFAR100,
    }[dataset]
    train_data = dataset_func(
        root=data_path,
        train=True,
        download=True,
        transform=train_transform)
    test_data = dataset_func(
        root=data_path,
        train=False,
        download=True,
        transform=test_transform)

    # Extract validation examples
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    split_idx = int(np.floor((1-val_ratio)*num_train))
    train_idxs, val_idxs = indices[:split_idx], indices[split_idx:]
    train_sampler = SubsetRandomSampler(train_idxs)
    val_sampler = SubsetRandomSampler(val_idxs)

    # Construct train/val/test dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False if val_ratio > 0 else True,
        pin_memory=True,
        sampler=train_sampler if val_ratio > 0 else None,
        num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        pin_memory=True,
        sampler=val_sampler,
        num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers)
    if val_ratio == 0:
        val_loader = test_loader
    return train_loader, val_loader, test_loader


def get_cifar10(data_path, batch_size, num_workers, val_ratio, random_seed):
    return get_cifar('cifar10', data_path, batch_size, num_workers, val_ratio, random_seed)


def get_cifar100(data_path, batch_size, num_workers, val_ratio, random_seed):
    return get_cifar('cifar100', data_path, batch_size, num_workers, val_ratio, random_seed)


def get_imagenette(data_path, batch_size, num_workers, val_ratio, random_seed):
    """Load Imagenette data into train/val/test data loader"""

    input_size = 224
    data_path = os.path.join(data_path, 'imagenette2-320/')

    transform = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
    ])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    train_data = datasets.ImageFolder(os.path.join(data_path, 'train'), transform=transform_train)
    test_data = datasets.ImageFolder(os.path.join(data_path, 'val'), transform=transform)

    # Extract validation examples
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    split_idx = int(np.floor((1-val_ratio)*num_train))
    train_idxs, val_idxs = indices[:split_idx], indices[split_idx:]
    train_sampler = SubsetRandomSampler(train_idxs)
    val_sampler = SubsetRandomSampler(val_idxs)

    # Construct train/val/test dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False if val_ratio > 0 else True,
        pin_memory=True,
        sampler=train_sampler if val_ratio > 0 else None,
        num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        pin_memory=True,
        sampler=val_sampler,
        num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers)
    if val_ratio == 0:
        val_loader = test_loader
    return train_loader, val_loader, test_loader


def get_num_steps_per_epoch(config):
    # Compute number of train samples
    num_tr_samples_tot = {
        'mnist': 60000,
        'cifar10': 50000,
        'cifar100': 50000,
        'imagenette': 9469,
    }[config['meta']['dataset']]
    tr_ratio = 1 - config['train']['val_ratio']
    num_tr_samples = num_tr_samples_tot*tr_ratio

    # Return number of update steps per epoch
    return int(np.ceil(num_tr_samples//config['train']['batch_size']))
