"""
cls_token.py
------------
CLS Token module for FT-Transformer.

Author: Asal Javadpour
Project: XTab_Project
"""

import torch
import torch.nn as nn


class CLSToken(nn.Module):
    """
    Adds a learnable CLS token to the beginning of the token sequence.
    """

    def __init__(self, embedding_dim: int):
        super().__init__()

        self.cls = nn.Parameter(
            torch.randn(1, 1, embedding_dim)
        )

    def forward(self, x: torch.Tensor):

        batch_size = x.size(0)

        cls_token = self.cls.expand(
            batch_size,
            -1,
            -1
        )

        x = torch.cat(
            [cls_token, x],
            dim=1
        )

        return x