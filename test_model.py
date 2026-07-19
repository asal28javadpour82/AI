"""
test_model.py
-------------
Quick model forward-pass test.

Author: Asal Javadpour
Project: XTab_Project
"""

import torch

from configs.config import Config
from models.transformer import FTTransformer
from utils.dataset import AdultDataset
from utils.preprocessing import DataPreprocessor
from utils.dataloader import create_dataloaders


# ==========================================
# Load Dataset
# ==========================================

dataset = AdultDataset(
    "data/raw/adult.data"
)

dataframe = dataset.load()


# ==========================================
# Preprocessing
# ==========================================

preprocessor = DataPreprocessor()

preprocessor.detect_feature_types(dataframe)

dataframe = preprocessor.encode_categorical_features(dataframe)

dataframe = preprocessor.scale_numeric_features(dataframe)

X, y = preprocessor.split_features_target(dataframe)


# ==========================================
# DataLoader
# ==========================================

train_loader, _, _ = create_dataloaders(
    X,
    y,
    preprocessor.numeric_columns,
    preprocessor.categorical_columns,
)


# ==========================================
# Model
# ==========================================

model = FTTransformer()

model.eval()


# ==========================================
# Forward Pass Test
# ==========================================

x_num, x_cat, labels = next(iter(train_loader))

with torch.no_grad():

    outputs = model(
        x_numeric=x_num,
        x_categorical=x_cat,
    )

print("Numerical Input Shape :", x_num.shape)
print("Categorical Input Shape :", x_cat.shape)
print("Output Shape :", outputs.shape)
print("Labels Shape :", labels.shape)

assert outputs.shape[0] == labels.shape[0]
assert outputs.shape[1] == Config.NUM_CLASSES

print("\n✅ Model Forward Pass Successful!")