"""
Dataset Module
--------------
Load Adult Income dataset from local file and prepare train/validation/test splits.

Author: Asal Javadpour
Project: XTab-Lite
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


class AdultDataset:
    """
    Adult Income Dataset Loader
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
        "income"
    ]

    def __init__(self, data_path: str):

        self.data_path = Path(data_path)

        self.dataframe = None

    def load(self):

        self.dataframe = pd.read_csv(

            self.data_path,

            header=None,

            names=self.COLUMN_NAMES,

            na_values=" ?",

            skipinitialspace=True

        )

        return self.dataframe

    def remove_missing_values(self):

        self.dataframe = self.dataframe.dropna()

        return self.dataframe

    def split(self):

        train_df, temp_df = train_test_split(

            self.dataframe,

            test_size=0.30,

            random_state=42,

            stratify=self.dataframe["income"]

        )

        valid_df, test_df = train_test_split(

            temp_df,

            test_size=0.50,

            random_state=42,

            stratify=temp_df["income"]

        )

        return train_df, valid_df, test_df