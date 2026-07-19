import torch

from configs.config import Config
from models.head import ClassificationHead

model = ClassificationHead(
    embedding_dim=Config.EMBEDDING_DIM,
    num_classes=Config.NUM_CLASSES,
    dropout=Config.DROPOUT,
)

x = torch.randn(
    32,
    Config.EMBEDDING_DIM,
)

y = model(x)

print("Input Shape :", x.shape)
print("Output Shape:", y.shape)