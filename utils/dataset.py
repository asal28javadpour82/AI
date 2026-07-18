"""
Dataset Loader
--------------
Loads, preprocesses and prepares tabular datasets.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class TabularDataset:

    def __init__(self, csv_path, target_column):

        self.csv_path = csv_path
        self.target_column = target_column

        self.data = None

        self.X = None
        self.y = None

    def load_data(self):

        self.data = pd.read_csv(self.csv_path)

        return self.data

    def preprocess(self):

        df = self.data.copy()

        # Encode categorical columns
        for column in df.columns:

            if df[column].dtype == "object":

                encoder = LabelEncoder()

                df[column] = encoder.fit_transform(df[column])

        self.X = df.drop(columns=[self.target_column])

        self.y = df[self.target_column]

        return self.X, self.y

    def split(self):

        X_train, X_test, y_train, y_test = train_test_split(

            self.X,
            self.y,
            test_size=0.30,
            random_state=42,
            shuffle=True

        )

        X_valid, X_test, y_valid, y_test = train_test_split(

            X_test,
            y_test,
            test_size=0.50,
            random_state=42

        )

        return (

            X_train,
            X_valid,
            X_test,

            y_train,
            y_valid,
            y_test

        )