import torch

from configs.config import Config
from models.transformer_block import TransformerBlock

model = TransformerBlock(
    embedding_dim=Config.EMBEDDING_DIM,
    num_heads=Config.NUM_HEADS,
    ffn_multiplier=Config.FFN_MULTIPLIER,
    dropout=Config.DROPOUT,
)

x = torch.randn(
    16,
    15,
    Config.EMBEDDING_DIM,
)

y = model(x)

print("Input Shape :", x.shape)
print("Output Shape:", y.shape)