"""
dataloader.py
-------------
Creates PyTorch DataLoaders for FT-Transformer.

Author: Asal Javadpour
Project: XTab_Project
"""

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from configs.config import Config
from utils.torch_dataset import TabularDataset


def create_dataloaders(
    X,
    y,
    numerical_columns,
    categorical_columns,
):
    """
    Creates train, validation and test DataLoaders.

    Parameters
    ----------
    X : pandas.DataFrame
        Input features.

    y : pandas.Series
        Target labels.

    numerical_columns : list
        Numerical feature names.

    categorical_columns : list
        Categorical feature names.

    Returns
    -------
    train_loader
    valid_loader
    test_loader
    """

    # ==========================================
    # Train / Test Split
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_SEED,
        stratify=y,
    )

    # ==========================================
    # Train / Validation Split
    # ==========================================

    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=Config.VALIDATION_SIZE,
        random_state=Config.RANDOM_SEED,
        stratify=y_train,
    )

    # ==========================================
    # Datasets
    # ==========================================

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

    # ==========================================
    # DataLoaders
    # ==========================================

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return (
        train_loader,
        valid_loader,
        test_loader,
    )