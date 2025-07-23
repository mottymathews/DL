import argparse
import time as time_module
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from homework.models import Classifier, save_model
from homework.datasets.classification_dataset import load_data
from homework.metrics import AccuracyMetric


def train(
    exp_dir: str = "logs",
    model_name: str = "classifier",
    num_epoch: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 2024,
    transform_pipeline: str = "default",
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        print("CUDA not available, using CPU")
        device = torch.device("cpu")

    # Start timing
    start_time = time_module.time()

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Create model - using Classifier for homework3
    model = Classifier(in_channels=3, num_classes=6, **kwargs)
    model = model.to(device)
    model.train()

    # Load data - using homework3 classification dataset
    train_data = load_data("classification_data/train", 
                          transform_pipeline=transform_pipeline,
                          shuffle=True, 
                          batch_size=batch_size, 
                          num_workers=2)
    val_data = load_data("classification_data/val", 
                        transform_pipeline="default",  # No augmentation for validation
                        shuffle=False,
                        batch_size=batch_size,
                        num_workers=2)

    # create loss function and optimizer
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    
    # Use homework3 metrics
    train_accuracy_metric = AccuracyMetric()
    val_accuracy_metric = AccuracyMetric()
    
    best_val_acc = 0.0

    # training loop
    for epoch in range(num_epoch):
        # Reset metrics at beginning of epoch
        train_accuracy_metric.reset()
        val_accuracy_metric.reset()

        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            # Training step
            optimizer.zero_grad()  # clear gradients
            logits = model(img)  # forward pass
            loss = loss_func(logits, label)  # compute loss
            
            logger.add_scalar("train_loss", loss.item(), global_step=global_step)
            epoch_train_loss += loss.item()
            num_train_batches += 1

            # Calculate training accuracy using homework3 metrics
            pred = model.predict(img)  # Use model's predict method
            train_accuracy_metric.add(pred, label)

            loss.backward()  # backward pass
            optimizer.step()  # update weights

            global_step += 1

        # Validation phase
        with torch.inference_mode():
            model.eval()
            epoch_val_loss = 0.0
            num_val_batches = 0

            for img, label in val_data:
                img, label = img.to(device), label.to(device)

                # Validation step
                logits = model(img)
                val_loss = loss_func(logits, label)
                epoch_val_loss += val_loss.item()
                num_val_batches += 1
                
                # Calculate validation accuracy
                pred = model.predict(img)
                val_accuracy_metric.add(pred, label)

        # Compute epoch metrics
        train_metrics = train_accuracy_metric.compute()
        val_metrics = val_accuracy_metric.compute()
        epoch_train_acc = train_metrics["accuracy"]
        epoch_val_acc = val_metrics["accuracy"]
        avg_train_loss = epoch_train_loss / num_train_batches
        avg_val_loss = epoch_val_loss / num_val_batches

        # Log epoch metrics
        logger.add_scalar("epoch_train_acc", epoch_train_acc, global_step=epoch)
        logger.add_scalar("epoch_val_acc", epoch_val_acc, global_step=epoch)
        logger.add_scalar("epoch_train_loss", avg_train_loss, global_step=epoch)
        logger.add_scalar("epoch_val_loss", avg_val_loss, global_step=epoch)

        # Save best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            save_model(model)
            print(f"New best model saved! Val accuracy: {epoch_val_acc:.4f}")

        # Print progress every epoch
        print(
            f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
            f"train_loss={avg_train_loss:.4f} "
            f"train_acc={epoch_train_acc:.4f} "
            f"val_loss={avg_val_loss:.4f} "
            f"val_acc={epoch_val_acc:.4f}"
        )

    end_time = time_module.time()
    training_time = end_time - start_time
    
    print(f"\nTraining completed for {model_name}")
    print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Log final metrics
    logger.add_scalar("training/total_time_seconds", training_time, global_step=0)
    logger.add_scalar("training/time_per_epoch", training_time/num_epoch, global_step=0)
    logger.add_scalar("training/best_val_accuracy", best_val_acc, global_step=0)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")

    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, default="classifier")
    parser.add_argument("--num_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--transform_pipeline", type=str, default="default", 
                        help="Transform pipeline: 'default' or 'aug' for augmentation")

    # pass all arguments to train
    train(**vars(parser.parse_args()))
