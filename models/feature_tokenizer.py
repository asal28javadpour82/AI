"""
Feature Tokenizer
-----------------
Transforms numerical and categorical tabular features into embeddings
for FT-Transformer.

Author: Asal Javadpour
Project: XTab-Lite
"""

from typing import List

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):
    """
    Feature Tokenizer for FT-Transformer.
    """

    def __init__(
        self,
        num_numeric_features: int,
        categorical_cardinalities: List[int],
        embedding_dim: int
    ) -> None:

        super().__init__()

        self.num_numeric_features = num_numeric_features
        self.embedding_dim = embedding_dim

        # One learnable weight vector per numerical feature
        self.numeric_weight = nn.Parameter(
            torch.randn(num_numeric_features, embedding_dim)
        )

        self.numeric_bias = nn.Parameter(
            torch.randn(num_numeric_features, embedding_dim)
        )

        # Embedding layer for every categorical feature
        self.categorical_embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, embedding_dim)
                for cardinality in categorical_cardinalities
            ]
        )

    def forward(
        self,
        x_numeric: torch.Tensor,
        x_categorical: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x_numeric:
            Shape -> (batch_size, num_numeric_features)

        x_categorical:
            Shape -> (batch_size, num_categorical_features)

        Returns
        -------
        Tensor
            Shape -> (batch_size, total_tokens, embedding_dim)
        """

        # Numerical feature tokenization
        numeric_tokens = (
            x_numeric.unsqueeze(-1)
            * self.numeric_weight.unsqueeze(0)
            + self.numeric_bias.unsqueeze(0)
        )

        # Categorical feature tokenization
        categorical_tokens = []

        for i, embedding in enumerate(self.categorical_embeddings):

            categorical_tokens.append(
                embedding(x_categorical[:, i])
            )

        categorical_tokens = torch.stack(
            categorical_tokens,
            dim=1
        )

        # Concatenate all tokens
        tokens = torch.cat(
            [numeric_tokens, categorical_tokens],
            dim=1
        )

        return tokens