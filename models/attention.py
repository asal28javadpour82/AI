"""
attention.py
------------
Multi-Head Self-Attention module for FT-Transformer.

Author: Asal Javadpour
Project: XTab_Project
"""

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float,
    ):

        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        output, _ = self.attention(
            x,
            x,
            x,
            need_weights=False,
        )

        return output