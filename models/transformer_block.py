"""
transformer_block.py
--------------------
Transformer Block for FT-Transformer.

Author: Asal Javadpour
Project: XTab_Project
"""

import torch
import torch.nn as nn

from models.multihead_attention import MultiHeadSelfAttention
from models.feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    Pre-Norm Transformer Block.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        ffn_multiplier: int,
        dropout: float,
    ):

        super().__init__()

        # LayerNorm before Attention
        self.norm1 = nn.LayerNorm(
            embedding_dim
        )

        # Multi-Head Attention
        self.attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.dropout1 = nn.Dropout(
            dropout
        )

        # LayerNorm before FFN
        self.norm2 = nn.LayerNorm(
            embedding_dim
        )

        # Feed Forward
        self.ffn = FeedForward(
            embedding_dim=embedding_dim,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
        )

        self.dropout2 = nn.Dropout(
            dropout
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # Attention Block
        attention_output = self.attention(
            self.norm1(x)
        )

        x = x + self.dropout1(
            attention_output
        )

        # Feed Forward Block
        ffn_output = self.ffn(
            self.norm2(x)
        )

        x = x + self.dropout2(
            ffn_output
        )

        return x