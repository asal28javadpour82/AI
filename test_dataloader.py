from utils.dataset import AdultDataset
from utils.preprocessing import DataPreprocessor
from utils.dataloader import create_dataloaders

dataset = AdultDataset("data/raw/adult.data")

df = dataset.load()

df = dataset.remove_missing_values()

preprocessor = DataPreprocessor()

preprocessor.detect_feature_types(df)

df = preprocessor.encode_categorical_features(df)

df = preprocessor.scale_numeric_features(df)

X, y = preprocessor.split_features_target(df)

train_loader, valid_loader, test_loader = create_dataloaders(
    X,
    y,
    numerical_columns=preprocessor.numeric_columns,
    categorical_columns=preprocessor.categorical_columns,
)

print("Train Batches :", len(train_loader))
print("Validation Batches :", len(valid_loader))
print("Test Batches :", len(test_loader))

for x_num, x_cat, label in train_loader:

    print()

    print("Numerical :", x_num.shape)

    print("Categorical :", x_cat.shape)

    print("Label :", label.shape)

    break