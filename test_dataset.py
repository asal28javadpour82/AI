from utils.dataset import AdultDataset
from utils.preprocessing import DataPreprocessor
from utils.torch_dataset import TabularDataset

dataset = AdultDataset("data/raw/adult.data")

df = dataset.load()

df = dataset.remove_missing_values()

preprocessor = DataPreprocessor()

preprocessor.detect_feature_types(df)

df = preprocessor.encode_categorical_features(df)

df = preprocessor.scale_numeric_features(df)

X, y = preprocessor.split_features_target(df)

dataset = TabularDataset(
    X=X,
    y=y,
    numerical_columns=preprocessor.numeric_columns,
    categorical_columns=preprocessor.categorical_columns
)

print("Dataset Length:", len(dataset))

x_num, x_cat, label = dataset[0]

print()

print("Numerical Shape :", x_num.shape)

print("Categorical Shape :", x_cat.shape)

print("Label :", label)