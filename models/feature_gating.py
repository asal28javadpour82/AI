"""
feature_gating.py
-----------------
Adaptive Feature Gating Module.

Author: Asal Javadpour
Project: XTab_Project
"""

import torch
import torch.nn as nn


class FeatureGating(nn.Module):
    """
    Adaptive Feature Gating.

    Learns an importance weight for every feature token.
    """

    def __init__(self, embedding_dim: int):

        super().__init__()

        self.gate = nn.Sequential(

            nn.Linear(
                embedding_dim,
                embedding_dim,
            ),

            nn.Sigmoid(),

        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        weights = self.gate(x)

        x = x * weights

        return x