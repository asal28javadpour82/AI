"""
head.py
-------
Classification Head for FT-Transformer.

Author: Asal Javadpour
Project: XTab_Project
"""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Final classification head.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        dropout: float,
    ):

        super().__init__()

        self.head = nn.Sequential(

            nn.LayerNorm(
                embedding_dim
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                embedding_dim,
                num_classes,
            ),
        )

    def forward(
        self,
        cls_embedding: torch.Tensor,
    ) -> torch.Tensor:

        return self.head(cls_embedding)