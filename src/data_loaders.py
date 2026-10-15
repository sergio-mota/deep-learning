import logging

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_cifar10_loaders(
        root : str,
        batch_size : int,
        num_workers : int
    ) -> dict[str, DataLoader]:
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = CIFAR10(
        root,
        train=True,
        download=True,
        transform=transform
    )
    test_dataset = CIFAR10(
        root,
        train=False,
        download=True,
        transform=transform
    )
    logger.info(f"Dataset size: train {len(train_dataset)}, test {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers
    )

    return {
        "train": train_loader,
        "test": test_loader
    }
