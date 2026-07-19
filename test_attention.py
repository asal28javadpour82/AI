import torch

from configs.config import Config
from models.attention import MultiHeadSelfAttention

batch_size = 16

num_tokens = 15

embedding_dim = Config.EMBEDDING_DIM

model = MultiHeadSelfAttention(
    embedding_dim=embedding_dim,
    num_heads=Config.NUM_HEADS,
    dropout=Config.DROPOUT,
)

x = torch.randn(
    batch_size,
    num_tokens,
    embedding_dim,
)

y = model(x)

print("Input Shape :", x.shape)

print("Output Shape:", y.shape)