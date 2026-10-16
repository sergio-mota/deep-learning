import logging

import mlflow
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def report_train_statistics(
        loss : float, 
        writer : torch.utils.tensorboard.SummaryWriter,
        epoch_idx : int,
        batch_idx : int, 
        num_batches : int
    ):
    logger.info(f"Batch {batch_idx+1} loss: {loss:0.3f}")
    x = epoch_idx * num_batches + batch_idx
    writer.add_scalar("batch loss", loss, x)
    mlflow.log_metric("batch loss", loss, step=x)

def train(
        model : torch.nn.Module,
        optimizer : torch.optim.Optimizer,
        criterion : torch.nn.Module,
        train_loader : torch.utils.data.DataLoader,
        validation_loader : torch.utils.data.DataLoader,
        device : torch.device,
        writer : torch.utils.tensorboard.SummaryWriter,
        num_epochs : int,
        *,
        scheduler : torch.optim.lr_scheduler = None
    ):
    for epoch_idx in range(num_epochs):
        logger.info(f"Epoch {epoch_idx+1}/{num_epochs}")
        train_loss = train_epoch(model, optimizer, criterion, train_loader, device, writer, epoch_idx)
        validation_loss, validation_accuracy = validate(model, criterion, validation_loader, device)
        if scheduler is not None:
            scheduler.step()
        
        logger.info(
            f"train loss: {train_loss:0.3f}, "
            f"validation loss: {validation_loss:0.3f}, "
            f"validation accuracy: {validation_accuracy:0.3f}"
        )
        writer.add_scalars(
            "loss",
            {
                "train": train_loss,
                "validation": validation_loss
            },
            epoch_idx
        )
        writer.add_scalars(
            "accuracy",
            {
                "validation": validation_accuracy
            },
            epoch_idx
        )
        mlflow.log_metrics(
            {
                "loss/train": train_loss,
                "loss/validation": validation_loss,
                "accuracy/validation": validation_accuracy
            },
            step=epoch_idx
        )

def train_batch(
        model : torch.nn.Module,
        optimizer : torch.optim.Optimizer,
        criterion : torch.nn.Module,
        batch : tuple[torch.Tensor, torch.Tensor],
        device : torch.device
    ):
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    logger.debug(f"Training loss: {loss.item():0.3f}")

    return loss

def train_epoch(
        model : torch.nn.Module,
        optimizer : torch.optim.Optimizer,
        criterion : torch.nn.Module,
        data_loader : torch.utils.data.DataLoader,
        device : torch.device,
        writer : torch.utils.tensorboard.SummaryWriter,
        epoch_idx : int
    ):
    model.train()

    num_batches = len(data_loader)
    report_interval = num_batches // 4  # Report every 25% of the epoch

    epoch_loss = 0.
    running_loss = 0.
    for batch_idx, batch in enumerate(data_loader):
        batch_loss = train_batch(model, optimizer, criterion, batch, device).item()
        running_loss += batch_loss
        epoch_loss += batch_loss

        if report_interval > 0 and batch_idx % report_interval == report_interval - 1:
            report_train_statistics(running_loss / report_interval, writer, epoch_idx, batch_idx, num_batches)
            running_loss = 0.

    remaining = len(data_loader) % report_interval if report_interval > 0 else num_batches
    if remaining != 0:
        report_train_statistics(running_loss / remaining, writer, epoch_idx, num_batches - 1, num_batches)

    return epoch_loss / num_batches

def validate_batch(
        model : torch.nn.Module,
        criterion : torch.nn.Module,
        batch : tuple[torch.Tensor, torch.Tensor],
        device : torch.device
    ):
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)

    with torch.no_grad():
        outputs = model(inputs)
        loss = criterion(outputs, targets).item()
        predictions = torch.argmax(outputs, dim=1)
        num_correct = (predictions == targets).sum().item()

    logger.debug(f"Batch: loss: {loss:0.3f}, num_correct: {num_correct}")

    return loss, num_correct

def validate(
        model : torch.nn.Module,
        criterion : torch.nn.Module,
        data_loader : torch.utils.data.DataLoader,
        device : torch.device
    ):
    model.eval()

    loss = 0.
    num_correct = 0
    total = 0
    for batch in data_loader:
        batch_loss, batch_num_correct = validate_batch(model, criterion, batch, device)
        loss += batch_loss
        num_correct += batch_num_correct
        total += len(batch[0])

    num_batches = len(data_loader)
    loss = loss / num_batches if num_batches > 0 else 0.
    accuracy = num_correct / total if total > 0 else 0.
    logger.info(f"Validation: loss: {loss:0.3f}, accuracy: {accuracy:0.3f}")

    return loss, accuracy
