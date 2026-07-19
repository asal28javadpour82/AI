import torch

from models.feature_tokenizer import FeatureTokenizer

batch_size = 32

num_numeric = 6

categorical_cardinalities = [
    10,
    8,
    15,
    6,
    12,
    5,
    4,
    9,
]

embedding_dim = 32

model = FeatureTokenizer(
    num_numeric_features=num_numeric,
    categorical_cardinalities=categorical_cardinalities,
    embedding_dim=embedding_dim,
)

x_numeric = torch.randn(batch_size, num_numeric)

x_categorical = torch.randint(
    0,
    4,
    (
        batch_size,
        len(categorical_cardinalities),
    ),
)

tokens = model(
    x_numeric,
    x_categorical,
)

print(tokens.shape)