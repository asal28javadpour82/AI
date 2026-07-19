"""
torch_dataset.py
----------------
PyTorch Dataset for FT-Transformer

Author: Asal Javadpour
Project: XTab_Project
"""

from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset


class TabularDataset(Dataset):
    """
    PyTorch Dataset for tabular data.
    Separates numerical and categorical features.
    """

    def __init__(
        self,
        X,
        y,
        numerical_columns: List[str],
        categorical_columns: List[str]
    ):

        self.numerical_columns = numerical_columns
        self.categorical_columns = categorical_columns

        self.X_num = X[numerical_columns].values.astype(np.float32)

        self.X_cat = X[categorical_columns].values.astype(np.int64)

        self.y = y.values.astype(np.int64)

    def __len__(self):

        return len(self.y)

    def __getitem__(self, idx):

        x_num = torch.tensor(self.X_num[idx], dtype=torch.float32)

        x_cat = torch.tensor(self.X_cat[idx], dtype=torch.long)

        label = torch.tensor(self.y[idx], dtype=torch.long)

        return x_num, x_cat, label