import torch

from models.feature_tokenizer import FeatureTokenizer

batch_size = 8

num_numeric_features = 6

categorical_cardinalities = [8, 16, 5]

embedding_dim = 32

model = FeatureTokenizer(
    num_numeric_features=num_numeric_features,
    categorical_cardinalities=categorical_cardinalities,
    embedding_dim=embedding_dim
)

x_numeric = torch.randn(batch_size, num_numeric_features)

x_categorical = torch.randint(
    0,
    5,
    (batch_size, len(categorical_cardinalities))
)

output = model(
    x_numeric,
    x_categorical
)

print(output.shape)