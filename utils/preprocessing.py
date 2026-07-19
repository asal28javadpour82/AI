"""
Preprocessing Module
--------------------
Data preprocessing for Adult Income Dataset.

Author: Asal Javadpour
Project: XTab_Project
"""

from typing import Tuple

import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


class DataPreprocessor:
    """
    Data preprocessing pipeline for FT-Transformer.
    """

    def __init__(self):

        # Store encoders for inference
        self.label_encoders = {}

        # Standard scaler for numerical features
        self.scaler = StandardScaler()

        # Feature lists
        self.categorical_columns = []
        self.numeric_columns = []

    def detect_feature_types(
        self,
        dataframe: pd.DataFrame
    ) -> None:
        """
        Detect numerical and categorical columns.
        """

        self.categorical_columns = list(
            dataframe.select_dtypes(include=["object"]).columns
        )

        self.numeric_columns = list(
            dataframe.select_dtypes(exclude=["object"]).columns
        )

        # Remove target column
        if "income" in self.categorical_columns:
            self.categorical_columns.remove("income")

        if "income" in self.numeric_columns:
            self.numeric_columns.remove("income")

    def encode_categorical_features(
        self,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Encode categorical columns using LabelEncoder.
        """

        df = dataframe.copy()

        for column in self.categorical_columns:

            encoder = LabelEncoder()

            df[column] = encoder.fit_transform(df[column])

            self.label_encoders[column] = encoder

        # Encode target
        target_encoder = LabelEncoder()

        df["income"] = target_encoder.fit_transform(df["income"])

        self.label_encoders["income"] = target_encoder

        return df

    def scale_numeric_features(
        self,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Standardize numerical columns.
        """

        df = dataframe.copy()

        df[self.numeric_columns] = self.scaler.fit_transform(
            df[self.numeric_columns]
        )

        return df

    def split_features_target(
        self,
        dataframe: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Split dataframe into features and target.
        """

        X = dataframe.drop(columns=["income"])

        y = dataframe["income"]

        return X, y

    def get_categorical_cardinalities(
        self,
        dataframe: pd.DataFrame
    ):
        """
        Return number of unique values for every categorical feature.

        This information is required by FT-Transformer
        to build categorical embeddings.
        """

        cardinalities = []

        for column in self.categorical_columns:
            cardinalities.append(
                dataframe[column].nunique()
            )

        return cardinalities