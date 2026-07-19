"""
multihead_attention.py
----------------------
Multi-Head Self-Attention for FT-Transformer.

Author: Asal Javadpour
Project: XTab_Project
"""

import math

import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention module.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        dropout: float
    ):

        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "embedding_dim must be divisible by num_heads."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        # Query
        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=True,
        )

        # Key
        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=True,
        )

        # Value
        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=True,
        )

        # Output projection
        self.output = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=True,
        )
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key.weight)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.xavier_uniform_(self.output.weight)

        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key.bias)
        nn.init.zeros_(self.value.bias)
        nn.init.zeros_(self.output.bias)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):

        batch_size, num_tokens, _ = x.size()

        x = x.view(
            batch_size,
            num_tokens,
            self.num_heads,
            self.head_dim,
        )

        x = x.transpose(1, 2)

        return x

    def combine_heads(self, x):

        batch_size, _, num_tokens, _ = x.size()

        x = x.transpose(1, 2).contiguous()

        x = x.view(
            batch_size,
            num_tokens,
            self.embedding_dim,
        )

        return x

    def forward(self, x):

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        attention_scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        )

        attention_scores = attention_scores / math.sqrt(
            self.head_dim
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1,
        )

        attention_weights = self.dropout(
            attention_weights
        )

        context = torch.matmul(
            attention_weights,
            v,
        )

        context = self.combine_heads(context)

        output = self.output(context)

        return output