"""
Training script for Homework 4 - Planners

Usage:
    python3 -m homework.train_planner --model mlp_planner --epochs 50 --batch_size 64
    python3 -m homework.train_planner --model transformer_planner --epochs 100 --batch_size 32
    python3 -m homework.train_planner --model cnn_planner --epochs 75 --batch_size 32
"""

import argparse
import time as time_module
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

from .models import MLPPlanner, TransformerPlanner, CNNPlanner, save_model
from .datasets.road_dataset import load_data
from .metrics import PlannerMetric


def train(
    model_name: str = "mlp_planner",
    exp_dir: str = "logs",
    num_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    seed: int = 2024,
    transform_pipeline: str = "default",
    **kwargs,
):
    """
    Training function for planner models
    
    Args:
        model_name: One of "mlp_planner", "transformer_planner", "cnn_planner"
        exp_dir: Directory to save logs and models
        num_epochs: Number of training epochs
        lr: Learning rate
        batch_size: Batch size
        seed: Random seed for reproducibility
        transform_pipeline: Data transform pipeline to use
    """
    
    # Device setup
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create log directory with timestamp
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # Model creation
    if model_name == "mlp_planner":
        model = MLPPlanner(n_track=10, n_waypoints=3)
        # For MLP and Transformer, use state_only pipeline (no images)
        transform_pipeline = "state_only"
    elif model_name == "transformer_planner":
        model = TransformerPlanner(n_track=10, n_waypoints=3, d_model=64)
        transform_pipeline = "state_only"
    elif model_name == "cnn_planner":
        model = CNNPlanner(n_waypoints=3)
        # For CNN, use default pipeline (includes images)
        transform_pipeline = "default"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model = model.to(device)
    print(f"Created {model_name} model")

    # Record start time
    start_time = time_module.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"Training {model_name} for {num_epochs} epochs")
    print(f"Start time: {start_datetime}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Transform pipeline: {transform_pipeline}")
    print("-" * 50)

    # Data loading
    print("Loading data...")
    train_data = load_data(
        "drive_data/train",
        transform_pipeline=transform_pipeline,
        shuffle=True,
        batch_size=batch_size,
        num_workers=2
    )
    
    val_data = load_data(
        "drive_data/val",
        transform_pipeline=transform_pipeline,
        shuffle=False,
        batch_size=batch_size,
        num_workers=2
    )

    # Loss function - MSE for regression task
    loss_fn = nn.MSELoss()

    # Optimizer - AdamW with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Metrics
    train_metric = PlannerMetric()
    val_metric = PlannerMetric()

    # Training state
    best_val_loss = float('inf')
    global_step = 0
    start_time = time_module.time()

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        # Reset metrics
        train_metric.reset()
        val_metric.reset()

        # Training phase
        model.train()
        epoch_train_loss = 0.0
        num_train_batches = 0

        for batch in train_data:
            # Move data to device
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}

            optimizer.zero_grad()

            # Forward pass - different inputs for different models
            if model_name in ["mlp_planner", "transformer_planner"]:
                # These models take track boundaries as input
                predictions = model(
                    track_left=batch["track_left"],
                    track_right=batch["track_right"]
                )
            else:  # cnn_planner
                # CNN model takes image as input
                predictions = model(image=batch["image"])

            # Get targets
            targets = batch["waypoints"]
            mask = batch["waypoints_mask"]

            # Compute loss only on valid waypoints
            loss = compute_masked_loss(loss_fn, predictions, targets, mask)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Update metrics
            epoch_train_loss += loss.item()
            num_train_batches += 1
            train_metric.add(predictions, targets, mask)

            # Log training loss
            logger.add_scalar("train/loss", loss.item(), global_step)
            global_step += 1

        # Validation phase
        model.eval()
        epoch_val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in val_data:
                # Move data to device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}

                # Forward pass
                if model_name in ["mlp_planner", "transformer_planner"]:
                    predictions = model(
                        track_left=batch["track_left"],
                        track_right=batch["track_right"]
                    )
                else:  # cnn_planner
                    predictions = model(image=batch["image"])

                # Get targets
                targets = batch["waypoints"]
                mask = batch["waypoints_mask"]

                # Compute loss
                loss = compute_masked_loss(loss_fn, predictions, targets, mask)

                # Update metrics
                epoch_val_loss += loss.item()
                num_val_batches += 1
                val_metric.add(predictions, targets, mask)

        # Compute epoch metrics
        avg_train_loss = epoch_train_loss / num_train_batches
        avg_val_loss = epoch_val_loss / num_val_batches

        train_metrics = train_metric.compute()
        val_metrics = val_metric.compute()

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(model)
            print(f"New best model saved! Val Loss: {avg_val_loss:.6f}")

        # Log epoch metrics
        logger.add_scalar("epoch/train_loss", avg_train_loss, epoch)
        logger.add_scalar("epoch/val_loss", avg_val_loss, epoch)
        logger.add_scalar("epoch/train_longitudinal_error", train_metrics["longitudinal_error"], epoch)
        logger.add_scalar("epoch/train_lateral_error", train_metrics["lateral_error"], epoch)
        logger.add_scalar("epoch/val_longitudinal_error", val_metrics["longitudinal_error"], epoch)
        logger.add_scalar("epoch/val_lateral_error", val_metrics["lateral_error"], epoch)

        # Print progress
        print(
            f"Epoch {epoch + 1:3d}/{num_epochs}: "
            f"train_loss={avg_train_loss:.6f} "
            f"val_loss={avg_val_loss:.6f} "
            f"train_long_err={train_metrics['longitudinal_error']:.4f} "
            f"train_lat_err={train_metrics['lateral_error']:.4f} "
            f"val_long_err={val_metrics['longitudinal_error']:.4f} "
            f"val_lat_err={val_metrics['lateral_error']:.4f}"
        )

    # Training completed
    end_time = time_module.time()
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    training_time = end_time - start_time

    # Format training time nicely
    hours = int(training_time // 3600)
    minutes = int((training_time % 3600) // 60)
    seconds = int(training_time % 60)
    
    print(f"\nTraining completed for {model_name}")
    print(f"End time: {end_datetime}")
    if hours > 0:
        print(f"Total training time: {hours}h {minutes}m {seconds}s ({training_time:.2f} seconds)")
    elif minutes > 0:
        print(f"Total training time: {minutes}m {seconds}s ({training_time:.2f} seconds)")
    else:
        print(f"Total training time: {seconds}s ({training_time:.2f} seconds)")
    print(f"Average time per epoch: {training_time/num_epochs:.2f} seconds")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Log final metrics
    logger.add_scalar("training/total_time_seconds", training_time)
    logger.add_scalar("training/time_per_epoch", training_time/num_epochs)
    logger.add_scalar("training/best_val_loss", best_val_loss)

    # Save final model
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Final model saved to {log_dir / f'{model_name}.th'}")

    logger.close()

    return model


def compute_masked_loss(loss_fn, predictions, targets, mask):
    """
    Compute loss only on valid waypoints using the mask
    
    Args:
        loss_fn: Loss function (e.g., MSELoss)
        predictions: (B, n_waypoints, 2)
        targets: (B, n_waypoints, 2)
        mask: (B, n_waypoints) boolean mask
    
    Returns:
        Masked loss value
    """
    # Expand mask to match prediction/target dimensions
    mask_expanded = mask.unsqueeze(-1).expand_as(predictions)  # (B, n_waypoints, 2)
    
    # Apply mask to predictions and targets
    masked_predictions = predictions[mask_expanded]
    masked_targets = targets[mask_expanded]
    
    # Compute loss only on valid waypoints
    if masked_predictions.numel() > 0:
        return loss_fn(masked_predictions, masked_targets)
    else:
        # If no valid waypoints, return zero loss
        return torch.tensor(0.0, device=predictions.device, requires_grad=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train planner models")
    
    parser.add_argument("--model", type=str, default="mlp_planner",
                        choices=["mlp_planner", "transformer_planner", "cnn_planner"],
                        help="Model to train")
    parser.add_argument("--exp_dir", type=str, default="logs",
                        help="Experiment directory")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--seed", type=int, default=2024,
                        help="Random seed")

    args = parser.parse_args()
    
    # Train the model
    train(
        model_name=args.model,
        exp_dir=args.exp_dir,
        num_epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed
    )
