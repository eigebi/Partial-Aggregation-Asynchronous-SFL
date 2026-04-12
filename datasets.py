# datasets.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


def build_transforms(name: str):
    name = name.lower()
    if name == "cifar10":
        size = 32
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif name == "cifar100":
        size = 32
        mean = (0.5071, 0.4867, 0.4409)
        std = (0.2675, 0.2565, 0.2761)
    elif name == "ham10000":
        size = 224
        mean = (0.7635, 0.5479, 0.5759)
        std = (0.1404, 0.1582, 0.1628)
    
    train_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomCrop(size, padding=size//8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, test_tf



def get_datasets(name: str, root: str, valid_ratio: float = 0.0):
    name = name.lower()
    train_tf, test_tf = build_transforms(name)

    if name == "cifar10":
        train = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
        test = datasets.CIFAR10(root, train=False, download=True, transform=test_tf)
    elif name == "cifar100":
        train = datasets.CIFAR100(root, train=True, download=True, transform=train_tf)
        test = datasets.CIFAR100(root, train=False, download=True, transform=test_tf)
    elif name == "ham10000":
        #train_full = datasets.HAM10000(root, train=True, download=True, transform=train_tf)
        #test = datasets.HAM10000(root, train=False, download=True, transform=test_tf)
        raise NotImplementedError("HAM10000 dataset is not implemented yet.")
    else:
        raise ValueError(f"Unsupported dataset: '{name}'.")

    return train, test

if __name__ == "__main__":
    train_set, test_set = get_datasets("cifar10", "./data")
    print("Train size:", len(train_set))
    print("Test size:", len(test_set))
    x, y = train_set[0]
    print("Sample x shape:", x.shape, "label:", y)