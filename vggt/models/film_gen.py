import torch
import torch.nn as nn
import torch.nn.functional as F

class FiLM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, feature_dim: int):
        """
        Args:
            input_dim (int): Dimensionality of metadata (e.g., 6)
            hidden_dim (int): Hidden dim for MLP
            feature_dim (int): Dim of the features to modulate (e.g., ViT dim)
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2 * feature_dim)  # γ and β
        )

    def forward(self, features: torch.Tensor, metadata: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, N, D) — D is feature_dim (from ViT or transformer)
            metadata: (B, 6) — latitude, longitude, altitude, pitch, roll, yaw
        Returns:
            modulated features: (B, N, D)
        """
        gamma_beta = self.mlp(metadata)  # (B, 2*D)
        gamma, beta = torch.chunk(gamma_beta, chunks=2, dim=-1)  # each (B, D)

        # Expand for broadcasting over tokens
        gamma = gamma.unsqueeze(1)  # (B, 1, D)
        beta = beta.unsqueeze(1)    # (B, 1, D)

        return gamma * features + beta
