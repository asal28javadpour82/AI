from utils.dataset import AdultDataset

dataset = AdultDataset("data/raw/adult.data")

df = dataset.load()

df = dataset.remove_missing_values()

train_df, valid_df, test_df = dataset.split()

print("=" * 50)

print("Train Shape :", train_df.shape)

print("Validation Shape :", valid_df.shape)

print("Test Shape :", test_df.shape)

print("=" * 50)

print(df.head())