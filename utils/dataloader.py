"""
dataloader.py
-------------
Creates PyTorch DataLoaders for FT-Transformer.

Author: Asal Javadpour
Project: XTab_Project
"""

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from utils.torch_dataset import TabularDataset


def create_dataloaders(
    X,
    y,
    numerical_columns,
    categorical_columns,
    batch_size=256,
    test_size=0.2,
    validation_size=0.1,
    random_state=42,
):
    """
    Create train, validation and test DataLoaders.
    """

    # ---------- Train / Test ----------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # ---------- Train / Validation ----------

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=validation_size,
        random_state=random_state,
        stratify=y_train,
    )

    train_dataset = TabularDataset(
        X_train,
        y_train,
        numerical_columns,
        categorical_columns,
    )

    valid_dataset = TabularDataset(
        X_valid,
        y_valid,
        numerical_columns,
        categorical_columns,
    )

    test_dataset = TabularDataset(
        X_test,
        y_test,
        numerical_columns,
        categorical_columns,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        valid_loader,
        test_loader,
    )