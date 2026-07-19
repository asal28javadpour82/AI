"""
feedforward.py
--------------
Feed Forward Network used in FT-Transformer.

Author: Asal Javadpour
Project: XTab_Project
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise Feed Forward Network.
    """

    def __init__(
        self,
        embedding_dim: int,
        ffn_multiplier: int,
        dropout: float,
    ):

        super().__init__()

        hidden_dim = embedding_dim * ffn_multiplier

        self.network = nn.Sequential(

            nn.Linear(
                embedding_dim,
                hidden_dim,
            ),

            nn.GELU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                embedding_dim,
            ),

            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        return self.network(x)