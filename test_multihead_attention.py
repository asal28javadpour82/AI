import torch

from configs.config import Config
from models.multihead_attention import MultiHeadSelfAttention

model = MultiHeadSelfAttention(
    embedding_dim=Config.EMBEDDING_DIM,
    num_heads=Config.NUM_HEADS,
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