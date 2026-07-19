"""
feature_tokenizer.py
--------------------
Feature Tokenizer for FT-Transformer.

Author: Asal Javadpour
Project: XTab_Project
"""

from typing import List

import torch
import torch.nn as nn


class FeatureTokenizer(nn.Module):

    def __init__(
        self,
        num_numeric_features: int,
        categorical_cardinalities: List[int],
        embedding_dim: int,
    ):

        super().__init__()

        self.num_numeric_features = num_numeric_features
        self.embedding_dim = embedding_dim

        # Numerical features
        self.numeric_weight = nn.Parameter(
            torch.empty(num_numeric_features, embedding_dim)
        )

        self.numeric_bias = nn.Parameter(
            torch.empty(num_numeric_features, embedding_dim)
        )

        nn.init.xavier_uniform_(self.numeric_weight)
        nn.init.zeros_(self.numeric_bias)

        # Categorical embeddings
        self.categorical_embeddings = nn.ModuleList()

        for cardinality in categorical_cardinalities:

            embedding = nn.Embedding(
                num_embeddings=cardinality,
                embedding_dim=embedding_dim,
            )

            nn.init.xavier_uniform_(embedding.weight)

            self.categorical_embeddings.append(embedding)

    def forward(
        self,
        x_numeric: torch.Tensor,
        x_categorical: torch.Tensor,
    ):

        numeric_tokens = (
            x_numeric.unsqueeze(-1)
            * self.numeric_weight.unsqueeze(0)
            + self.numeric_bias.unsqueeze(0)
        )

        categorical_tokens = []

        for i, embedding in enumerate(self.categorical_embeddings):

            categorical_tokens.append(
                embedding(x_categorical[:, i])
            )

        categorical_tokens = torch.stack(
            categorical_tokens,
            dim=1,
        )

        tokens = torch.cat(
            [numeric_tokens, categorical_tokens],
            dim=1,
        )

        return tokens