"""
transformer.py
--------------
Main FT-Transformer Model.

Author: Asal Javadpour
Project: XTab_Project
"""

import torch
import torch.nn as nn

from configs.config import Config
from models.feature_gating import FeatureGating
from models.feature_tokenizer import FeatureTokenizer
from models.cls_token import CLSToken
from models.transformer_block import TransformerBlock
from models.head import ClassificationHead


class FTTransformer(nn.Module):
    """
    FT-Transformer Model.
    """

    def __init__(self):

        super().__init__()

        # Feature Tokenizer
        self.feature_tokenizer = FeatureTokenizer(
            num_numeric_features=Config.NUM_NUMERIC_FEATURES,
            categorical_cardinalities=Config.CATEGORICAL_CARDINALITIES,
            embedding_dim=Config.EMBEDDING_DIM,
        )
        # ======================================
        # Adaptive Feature Gating
        # ======================================
        self.feature_gating = FeatureGating(
                embedding_dim=Config.EMBEDDING_DIM
        )

        # CLS Token
        self.cls_token = CLSToken(
            embedding_dim=Config.EMBEDDING_DIM
        )

        # Transformer Encoder
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=Config.EMBEDDING_DIM,
                    num_heads=Config.NUM_HEADS,
                    ffn_multiplier=Config.FFN_MULTIPLIER,
                    dropout=Config.DROPOUT,
                )
                for _ in range(Config.NUM_TRANSFORMER_BLOCKS)
            ]
        )

        # Final LayerNorm
        self.final_norm = nn.LayerNorm(
            Config.EMBEDDING_DIM
        )

        # Classification Head
        self.head = ClassificationHead(
            embedding_dim=Config.EMBEDDING_DIM,
            num_classes=Config.NUM_CLASSES,
            dropout=Config.DROPOUT,
        )

    def forward(
        self,
        x_numeric: torch.Tensor,
        x_categorical: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of FT-Transformer.
        """

        # ======================================
        # Feature Tokenization
        # ======================================

        x = self.feature_tokenizer(
            x_numeric=x_numeric,
            x_categorical=x_categorical,
        )

        # ======================================
        # Adaptive Feature Gating
        # ======================================

        x = self.feature_gating(x)

        # ======================================
        # CLS Token
        # ======================================

        x = self.cls_token(x)

        # ======================================
        # Transformer Encoder
        # ======================================

        for block in self.transformer_blocks:
            x = block(x)

        # ======================================
        # Final Layer Normalization
        # ======================================

        x = self.final_norm(x)

        cls_embedding = x[:, 0]

        output = self.head(cls_embedding)

        return output