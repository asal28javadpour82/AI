import torch

from configs.config import Config
from models.ffn import FeedForwardNetwork

model = FeedForwardNetwork(
    embedding_dim=Config.EMBEDDING_DIM,
    ffn_multiplier=Config.FFN_MULTIPLIER,
    dropout=Config.DROPOUT,
)

x = torch.randn(
    32,
    15,
    Config.EMBEDDING_DIM,
)

y = model(x)

print("Input Shape :", x.shape)
print("Output Shape:", y.shape)