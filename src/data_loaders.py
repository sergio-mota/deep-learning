import logging

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_cifar10_loaders(
        root : str,
        batch_size : int,
        num_workers : int
    ) -> dict[str, DataLoader]:
    train_transform = v2.Compose([
            v2.ToImage(),
            v2.RandomCrop(32, padding=4),
            v2.RandomHorizontalFlip(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    test_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    train_dataset = CIFAR10(
        root,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = CIFAR10(
        root,
        train=False,
        download=True,
        transform=test_transform
    )
    validation_dataset, test_dataset = random_split(test_dataset, [0.5, 0.5])
    logger.info(
        "Dataset size: "
        f"train {len(train_dataset)}, "
        f"valid {len(validation_dataset)}, "
        f"test {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    validation_loader = DataLoader(
        validation_dataset,
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
        "validation": validation_loader,
        "test": test_loader
    }
