"""
dataset.py
----------
Load Adult Income Dataset.

Author: Asal Javadpour
Project: XTab_Project
"""

from pathlib import Path

import pandas as pd


class AdultDataset:
    """
    Adult Income Dataset Loader.
    """

    COLUMN_NAMES = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education_num",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital_gain",
        "capital_loss",
        "hours_per_week",
        "native_country",
        "income",
    ]

    def __init__(self, data_path: str):

        self.data_path = Path(data_path)

    def load(self) -> pd.DataFrame:
        """
        Load dataset from CSV file.
        """

        dataframe = pd.read_csv(
            self.data_path,
            header=None,
            names=self.COLUMN_NAMES,
            na_values=" ?",
            skipinitialspace=True,
        )

        dataframe = dataframe.dropna()

        dataframe.reset_index(
            drop=True,
            inplace=True,
        )

        return dataframe