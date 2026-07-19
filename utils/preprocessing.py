"""
Preprocessing Module
--------------------
Data preprocessing for Adult Income Dataset.

Author: Asal Javadpour
Project: XTab-Lite
"""

from typing import List, Tuple

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:

    def __init__(self):

        self.label_encoders = {}

        self.scaler = StandardScaler()

        self.categorical_columns = []

        self.numeric_columns = []

    def detect_feature_types(self, dataframe: pd.DataFrame) -> None:
        """
        Detect numerical and categorical columns.
        """

        self.categorical_columns = list(
            dataframe.select_dtypes(include=["object"]).columns
        )

        self.numeric_columns = list(
            dataframe.select_dtypes(exclude=["object"]).columns
        )

        if "income" in self.categorical_columns:
            self.categorical_columns.remove("income")

    def encode_categorical_features(
        self,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:

        df = dataframe.copy()

        for column in self.categorical_columns:

            encoder = LabelEncoder()

            df[column] = encoder.fit_transform(df[column])

            self.label_encoders[column] = encoder

        target_encoder = LabelEncoder()

        df["income"] = target_encoder.fit_transform(df["income"])

        self.label_encoders["income"] = target_encoder

        return df

    def scale_numeric_features(
        self,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:

        df = dataframe.copy()

        self.numeric_columns.remove("income") if "income" in self.numeric_columns else None

        df[self.numeric_columns] = self.scaler.fit_transform(
            df[self.numeric_columns]
        )

        return df

    def split_features_target(
        self,
        dataframe: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:

        X = dataframe.drop(columns=["income"])

        y = dataframe["income"]

        return X, y