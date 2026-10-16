import logging
from datetime import datetime

import mlflow
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

from src.data_loaders import get_cifar10_loaders as get_loaders
from src.models import ResNet18_Cifar as Model
from src.trainer import train, validate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup(config_path):
    with open(config_path, "r") as f:
        try:
            logger.info(f"Loading configuration from {config_path}")
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            logger.error(exc)

    mlflow.set_experiment(config["experiment"]["name"])
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(1)

    torch.manual_seed(42)

    writer = SummaryWriter(f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    return config, writer

def get_criterion(hparams):
    if hparams["criterion"] == "CE":
        criterion = torch.nn.functional.cross_entropy
    else:
        raise ValueError(f"Unsupported criterion type: {hparams['criterion']}")

    return criterion

def get_optimizer(model, hparams):
    if hparams["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=hparams["learning_rate"],
            momentum=hparams["momentum"],
            weight_decay=hparams["weight_decay"]
        )
    else:
        raise ValueError(f"Unsupported optimizer type: {hparams['optimizer']}")

    return optimizer

def get_scheduler(optimizer, hparams):
    scheduler = None
    if hparams["scheduler"] == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            hparams["num_epochs"]
        )

    return scheduler

def main():
    config, writer = setup("config/cifar10.yaml")

    data_loaders = get_loaders(
        config["data"]["root"],
        config["hparams"]["batch_size"],
        config["data"]["num_workers"]
    )

    model = Model()
    optimizer = get_optimizer(model, config["hparams"])
    criterion = get_criterion(config["hparams"])
    scheduler = get_scheduler(optimizer, config["hparams"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.to(device)
    train(
        model,
        optimizer,
        criterion,
        data_loaders["train"],
        data_loaders["validation"],
        device,
        writer,
        config["hparams"]["num_epochs"],
        scheduler=scheduler
    )

    loss, accuracy = validate(model, criterion, data_loaders["test"], device)

if __name__ == "__main__":
    main()