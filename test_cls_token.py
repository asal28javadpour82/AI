import torch

from models.cls_token import CLSToken

batch_size = 32

num_tokens = 14

embedding_dim = 32

x = torch.randn(
    batch_size,
    num_tokens,
    embedding_dim
)

cls = CLSToken(embedding_dim)

output = cls(x)

print("Input Shape :", x.shape)

print("Output Shape:", output.shape)