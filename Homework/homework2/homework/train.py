import argparse
import time as time_module
from datetime import datetime, time
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import ClassificationLoss, load_model, save_model
from .utils import load_data


def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
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

    # note: the grader uses default kwargs, you'll have to bake them in for the final submission
    model = load_model(model_name, **kwargs)
    model = model.to(device)
    model.train()

    train_data = load_data("classification_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("classification_data/val", shuffle=False)

    # create loss function and optimizer
    loss_func = ClassificationLoss()
    # optimizer = ...
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            # TODO: implement training step
            optimizer.zero_grad()  # clear gradients
            logits = model(img)  # forward pass
            loss = loss_func(logits, label)  # compute loss
            logger.add_scalar("train_loss", loss.item(), global_step=global_step)

            # Calculate training accuracy
            pred = logits.argmax(dim=1)
            acc = (pred == label).float().mean()
            metrics["train_acc"].append(acc.item())  # ← SET HERE
            #logger.add_scalar("train_acc", acc.item(), global_step=global_step)

            loss.backward()  # backward pass
            optimizer.step()  # update weights
            #raise NotImplementedError("Training step not implemented")

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for img, label in val_data:
                img, label = img.to(device), label.to(device)

                # TODO: compute validation accuracy
                logits = model(img)
                preds = logits.argmax(dim=1)  # get predicted class
                acc = (preds == label).float().mean().item()  # compute accuracy
                metrics["val_acc"].append(acc)
                #logger.add_scalar("val_acc", acc, global_step=global_step)
                #raise NotImplementedError("Validation step not implemented")
    
        # Add this section for weight logging
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            # Log weight histograms
            for name, param in model.named_parameters():
                if param.grad is not None:
                    logger.add_histogram(f"weights/{name}", param.data, global_step=global_step)
                    logger.add_histogram(f"gradients/{name}", param.grad.data, global_step=global_step)
            
            # Log weight statistics
            total_params = sum(p.numel() for p in model.parameters())
            logger.add_scalar("model/total_parameters", total_params, global_step=global_step)

    end_time = time_module.time()
    training_time = end_time - start_time
    
    print(f"\nTraining completed for {model_name}")
    print(f"Total training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    
    # Log to TensorBoard
    if logger:
        logger.add_scalar("training/total_time_seconds", training_time, global_step=0)
        logger.add_scalar("training/time_per_epoch", training_time/num_epoch, global_step=0)
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        logger.add_scalar("epoch_train_acc", epoch_train_acc, global_step=global_step)
        logger.add_scalar("epoch_val_acc", epoch_val_acc, global_step=global_step)

        #raise NotImplementedError("Logging not implemented")

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024)

    # optional: additional model hyperparamters
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
