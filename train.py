"""
train.py
--------
Training script for FT-Transformer.

Author: Asal Javadpour
Project: XTab_Project
"""

from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from configs.config import Config

from models.transformer import FTTransformer

from utils.dataset import AdultDataset
from utils.preprocessing import DataPreprocessor
from utils.dataloader import create_dataloaders
from utils.visualization import plot_training_history
from evaluate import evaluate


# ==========================================
# Random Seed
# ==========================================

def set_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


set_seed(Config.RANDOM_SEED)


# ==========================================
# Device
# ==========================================

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)

print(f"Using Device : {device}")


# ==========================================
# Create Output Folder
# ==========================================

Path("outputs/checkpoints").mkdir(

    parents=True,

    exist_ok=True,

)


# ==========================================
# Load Dataset
# ==========================================

dataset = AdultDataset(

    "data/raw/adult.data"

)

dataframe = dataset.load()


# ==========================================
# Preprocessing
# ==========================================

preprocessor = DataPreprocessor()

preprocessor.detect_feature_types(

    dataframe

)

dataframe = preprocessor.encode_categorical_features(

    dataframe

)

dataframe = preprocessor.scale_numeric_features(

    dataframe

)

X, y = preprocessor.split_features_target(

    dataframe

)


# ==========================================
# DataLoaders
# ==========================================

train_loader, valid_loader, test_loader = create_dataloaders(

    X=X,

    y=y,

    numerical_columns=preprocessor.numeric_columns,

    categorical_columns=preprocessor.categorical_columns,

)


# ==========================================
# Model
# ==========================================

model = FTTransformer().to(device)


# ==========================================
# Loss
# ==========================================

criterion = nn.CrossEntropyLoss()


# ==========================================
# Optimizer
# ==========================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=Config.LEARNING_RATE,

    weight_decay=Config.WEIGHT_DECAY,

)


# ==========================================
# Scheduler
# ==========================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(

    optimizer,

    mode="min",

    factor=0.5,

    patience=3,

)
# ==========================================
# Train One Epoch
# ==========================================

def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
):

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    progress_bar = tqdm(

        dataloader,

        desc="Training",

        leave=False,

    )

    for x_num, x_cat, labels in progress_bar:

        x_num = x_num.to(device)

        x_cat = x_cat.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(

            x_numeric=x_num,

            x_categorical=x_cat,

        )

        loss = criterion(

            outputs,

            labels,

        )

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        predictions = torch.argmax(

            outputs,

            dim=1,

        )

        correct += (

            predictions == labels

        ).sum().item()

        total += labels.size(0)

        progress_bar.set_postfix(

            loss=f"{loss.item():.4f}"

        )

    epoch_loss = running_loss / len(

        dataloader

    )

    epoch_accuracy = (

        100.0 * correct / total

    )

    return epoch_loss, epoch_accuracy


# ==========================================
# Validation
# ==========================================

def validate(
    model,
    dataloader,
    criterion,
    device,
):

    model.eval()

    running_loss = 0.0

    correct = 0

    total = 0

    with torch.no_grad():

        for x_num, x_cat, labels in dataloader:

            x_num = x_num.to(device)

            x_cat = x_cat.to(device)

            labels = labels.to(device)

            outputs = model(

                x_numeric=x_num,

                x_categorical=x_cat,

            )

            loss = criterion(

                outputs,

                labels,

            )

            running_loss += loss.item()

            predictions = torch.argmax(

                outputs,

                dim=1,

            )

            correct += (

                predictions == labels

            ).sum().item()

            total += labels.size(0)

    validation_loss = running_loss / len(

        dataloader

    )

    validation_accuracy = (

        100.0 * correct / total

    )

    return validation_loss, validation_accuracy
# ==========================================
# Training Function
# ==========================================

def train():

    best_validation_loss = float("inf")

    early_stop_counter = 0

    train_losses = []

    validation_losses = []

    train_accuracies = []

    validation_accuracies = []

    for epoch in range(Config.EPOCHS):

        print("\n" + "=" * 60)

        print(f"Epoch [{epoch + 1}/{Config.EPOCHS}]")

        print("=" * 60)

        # --------------------------------------
        # Train
        # --------------------------------------

        train_loss, train_accuracy = train_one_epoch(

            model=model,

            dataloader=train_loader,

            criterion=criterion,

            optimizer=optimizer,

            device=device,

        )

        # --------------------------------------
        # Validation
        # --------------------------------------

        validation_loss, validation_accuracy = validate(

            model=model,

            dataloader=valid_loader,

            criterion=criterion,

            device=device,

        )

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

        train_accuracies.append(train_accuracy)
        validation_accuracies.append(validation_accuracy)

        scheduler.step(validation_loss)

        print(
            f"Train Loss : {train_loss:.4f} | "
            f"Train Accuracy : {train_accuracy:.2f}%"
        )

        print(
            f"Validation Loss : {validation_loss:.4f} | "
            f"Validation Accuracy : {validation_accuracy:.2f}%"
        )

        # --------------------------------------
        # Save Best Model
        # --------------------------------------

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            early_stop_counter = 0

            torch.save(

                model.state_dict(),

                Config.SAVE_MODEL_PATH,

            )

            print("\nBest Model Saved.")

        else:

            early_stop_counter += 1

            print(

                f"\nNo Improvement ({early_stop_counter}/{Config.PATIENCE})"

            )

        # --------------------------------------
        # Early Stopping
        # --------------------------------------

        if early_stop_counter >= Config.PATIENCE:

            print("\nEarly Stopping Activated.")

            break

        plot_training_history(
    train_losses=train_losses,
    validation_losses=validation_losses,
    train_accuracies=train_accuracies,
    validation_accuracies=validation_accuracies,
)
# ==========================================
# Test Best Model
# ==========================================

def test():

    print("\n" + "=" * 60)
    print("Loading Best Model ...")
    print("=" * 60)

    model.load_state_dict(

        torch.load(

            Config.SAVE_MODEL_PATH,

            map_location=device,

        )

    )

    results = evaluate(

        model=model,

        dataloader=test_loader,

        device=device,

    )

    print("\n")

    print("=" * 60)

    print("Final Test Results")

    print("=" * 60)

    print(

        f"Accuracy  : {results['accuracy']:.4f}"

    )

    print(

        f"Precision : {results['precision']:.4f}"

    )

    print(

        f"Recall    : {results['recall']:.4f}"

    )

    print(

        f"F1 Score  : {results['f1']:.4f}"

    )

    print("=" * 60)


# ==========================================
# Main
# ==========================================

def main():

    train()

    test()


if __name__ == "__main__":

    main()