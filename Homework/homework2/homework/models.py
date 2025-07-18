"""
Implement the following models for classification.

Feel free to modify the arguments for each of model's __init__ function.
This will be useful for tuning model hyperparameters such as hidden_dim, num_layers, etc,
but remember that the grader will assume the default constructor!
"""

from pathlib import Path

import torch
import torch.nn as nn


class ClassificationLoss(nn.Module):
    def forward(self, logits: torch.Tensor, target: torch.LongTensor) -> torch.Tensor:
        """
        Multi-class classification loss
        Hint: simple one-liner

        Args:
            logits: tensor (b, c) logits, where c is the number of classes
            target: tensor (b,) labels

        Returns:
            tensor, scalar loss
        """
        #print(f"Cross Entropy Loss = {nn.CrossEntropyLoss()(logits, target)}")
        return nn.CrossEntropyLoss()(logits, target)
    
        raise NotImplementedError("ClassificationLoss.forward() is not implemented")


class LinearClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()
        # A simple linear classifier
        # Hint: you can use nn.Linear to create a linear layer
        # The input size is 3 * h * w, and the output size is num_classes
        self.classifier = nn.Linear(3 * h * w, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten the input tensor and pass it through the linear layer
        x = x.view(x.size(0), -1)  # Flatten the input tensor
        logits = self.classifier(x)  # Pass through the linear layer
        #print(f"Input shape: {x.shape}")        # (3, 3, 64, 64)
        #print(f"Output shape: {logits.shape}")  # (3, 6)
        #print(f"Output:\n{logits}")
        return logits
        raise NotImplementedError("LinearClassifier.forward() is not implemented")


class MLPClassifier(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
    ):
        """
        An MLP with a single hidden layer

        Args:
            h: int, height of the input image
            w: int, width of the input image
            num_classes: int, number of classes
        """
        super().__init__()

        # A simple MLP with one hidden layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * h * w, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        #raise NotImplementedError("MLPClassifier.__init__() is not implemented")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        # Flatten the input tensor and pass it through the MLP
        x = self.classifier(x)
        return x
        raise NotImplementedError("MLPClassifier.forward() is not implemented")


class MLPClassifierDeep(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 4, 
        lr: float = 0.001, 
        batch_size: int = 512, 
        #dropout_rate: float = 0.3,  # ← Add dropout parameter
    ):
        
        """ 
        Total training time: 331.17 seconds (5.52 minutes)
        Epoch 50 / 50: train_acc=0.9843 val_acc=0.8080
        Model saved to logs/mlp_deep_0712_162646/mlp_deep.th
        
        def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 128,
        num_layers: int = 4, 
        lr: float = 0.001, 
        batch_size: int = 512, 
        #dropout_rate: float = 0.3,  # ← Add dropout parameter
    ):"""
        """
        An MLP with multiple hidden layers

        Args:
            h: int, height of image
            w: int, width of image
            num_classes: int

        Hint - you can add more arguments to the constructor such as:
            hidden_dim: int, size of hidden layers
            num_layers: int, number of hidden layers
        """
        super().__init__()
        # A deep MLP with multiple hidden layers
        layers = [nn.Flatten()]
        input_dim = 3 * h * w
        for _ in range(num_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

        #raise NotImplementedError("MLPClassifierDeep.__init__() is not implemented")
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: tensor (b, 3, H, W) image

        Returns:
            tensor (b, num_classes) logits
        """
        x = self.classifier(x)
        return x
        raise NotImplementedError("MLPClassifierDeep.forward() is not implemented")

class MLPClassifierDeepResidual(nn.Module):
    def __init__(
        self,
        h: int = 64,
        w: int = 64,
        num_classes: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 4,         # ← Reduce layers
        dropout_rate: float = 0.1,   # ← Strong dropout
    ):
        super().__init__()

        input_dim = 3 * h * w
        
        self.flatten = nn.Flatten()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = torch.relu(self.input_layer(x))
        x = self.dropout(x)
        
        # Apply residual with strong regularization
        for i, layer in enumerate(self.layers):
            residual = x
            x = layer(x)
            x = torch.relu(x + residual)
            # Apply dropout after every residual block
            x = self.dropout(x)
        
        return self.output_layer(x)


model_factory = {
    "linear": LinearClassifier,
    "mlp": MLPClassifier,
    "mlp_deep": MLPClassifierDeep,
    "mlp_deep_residual": MLPClassifierDeepResidual,
}


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Args:
        model: torch.nn.Module

    Returns:
        float, size in megabytes
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024


def save_model(model):
    """
    Use this function to save your model in train.py
    """
    for n, m in model_factory.items():
        if isinstance(model, m):
            return torch.save(model.state_dict(), Path(__file__).resolve().parent / f"{n}.th")
    raise ValueError(f"Model type '{str(type(model))}' not supported")


def load_model(model_name: str, with_weights: bool = False, **model_kwargs):
    """
    Called by the grader to load a pre-trained model by name
    """
    r = model_factory[model_name](**model_kwargs)
    if with_weights:
        model_path = Path(__file__).resolve().parent / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"
        try:
            r.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # Limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(r)
    if model_size_mb > 10:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")
    print(f"Model size: {model_size_mb:.2f} MB")

    return r
