from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Input: two tracks of n_track points each with 2 coordinates
        # Total input size: 2 * n_track * 2 = 4 * n_track
        input_size = 4 * n_track
        
        # Output: n_waypoints with 2 coordinates each
        output_size = n_waypoints * 2

        # MLP architecture
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, output_size),
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.shape[0]
        
        # Flatten and concatenate both tracks
        track_left_flat = track_left.view(batch_size, -1)  # (b, n_track * 2)
        track_right_flat = track_right.view(batch_size, -1)  # (b, n_track * 2)
        
        # Concatenate left and right tracks
        x = torch.cat([track_left_flat, track_right_flat], dim=1)  # (b, 4 * n_track)
        
        # Pass through MLP
        waypoints_flat = self.mlp(x)  # (b, n_waypoints * 2)
        
        # Reshape to waypoints format
        waypoints = waypoints_flat.view(batch_size, self.n_waypoints, 2)
        
        return waypoints


class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints
        self.d_model = d_model

        # Learned query embeddings for waypoints (Perceiver-style)
        self.query_embed = nn.Embedding(n_waypoints, d_model)
        
        # Input projection for track points
        self.input_projection = nn.Linear(2, d_model)
        
        # Positional encoding for track points
        self.pos_embed = nn.Parameter(torch.randn(2 * n_track, d_model))
        
        # Transformer decoder layers for cross-attention
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        
        # Output projection to waypoint coordinates
        self.output_projection = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        batch_size = track_left.shape[0]
        
        # Concatenate left and right tracks
        track_points = torch.cat([track_left, track_right], dim=1)  # (b, 2*n_track, 2)
        
        # Project track points to d_model dimensions
        track_features = self.input_projection(track_points)  # (b, 2*n_track, d_model)
        
        # Add positional encoding
        track_features = track_features + self.pos_embed.unsqueeze(0)  # (b, 2*n_track, d_model)
        
        # Get query embeddings for waypoints
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)  # (b, n_waypoints, d_model)
        
        # Apply transformer decoder (cross-attention)
        # queries attend to track_features
        waypoint_features = self.transformer_decoder(queries, track_features)  # (b, n_waypoints, d_model)
        
        # Project to waypoint coordinates
        waypoints = self.output_projection(waypoint_features)  # (b, n_waypoints, 2)
        
        return waypoints


class CNNPlanner(torch.nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
    ):
        super().__init__()

        self.n_waypoints = n_waypoints

        self.register_buffer("input_mean", torch.as_tensor(INPUT_MEAN), persistent=False)
        self.register_buffer("input_std", torch.as_tensor(INPUT_STD), persistent=False)

        # CNN backbone - similar to image classification models
        self.backbone = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 96x128 -> 48x64
            
            # Second conv block  
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 48x64 -> 24x32
            
            # Third conv block
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 24x32 -> 12x16
            
            # Fourth conv block
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 12x16 -> 6x8
        )
        
        # Calculate flattened size: 256 channels * 6 * 8 = 12288
        self.flattened_size = 256 * 6 * 8
        
        # Fully connected layers to predict waypoints
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, n_waypoints * 2),  # n_waypoints * 2 for (x, y) coordinates
        )

    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Args:
            image (torch.FloatTensor): shape (b, 3, h, w) and vals in [0, 1]

        Returns:
            torch.FloatTensor: future waypoints with shape (b, n, 2)
        """
        # Normalize input
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        
        # Pass through CNN backbone
        features = self.backbone(x)  # (b, 256, 6, 8)
        
        # Pass through prediction head
        waypoints_flat = self.head(features)  # (b, n_waypoints * 2)
        
        # Reshape to waypoint format
        batch_size = image.shape[0]
        waypoints = waypoints_flat.view(batch_size, self.n_waypoints, 2)
        
        return waypoints


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "cnn_planner": CNNPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
