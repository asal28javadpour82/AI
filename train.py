from utils.dataset import AdultDataset
from utils.preprocessing import DataPreprocessor

dataset = AdultDataset("data/raw/adult.data")

df = dataset.load()

df = dataset.remove_missing_values()

preprocessor = DataPreprocessor()

preprocessor.detect_feature_types(df)

df = preprocessor.encode_categorical_features(df)

df = preprocessor.scale_numeric_features(df)

X, y = preprocessor.split_features_target(df)

print("=" * 50)

print(X.head())

print("=" * 50)

print(y.head())