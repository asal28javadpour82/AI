"""
visualization.py
----------------
Plot training and validation metrics for FT-Transformer.

Author: Asal Javadpour
Project: XTab_Project
"""

from pathlib import Path

import matplotlib.pyplot as plt


def plot_training_history(
    train_losses,
    validation_losses,
    train_accuracies,
    validation_accuracies,
):
    """
    Plot loss and accuracy curves.
    """

    # Create output directory
    Path("outputs/figures").mkdir(
        parents=True,
        exist_ok=True,
    )

    epochs = range(
        1,
        len(train_losses) + 1,
    )

    # ======================================
    # Loss Curve
    # ======================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_losses,
        label="Train Loss",
        linewidth=2,
    )

    plt.plot(
        epochs,
        validation_losses,
        label="Validation Loss",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "outputs/figures/loss_curve.png"
    )

    plt.close()

    # ======================================
    # Accuracy Curve
    # ======================================

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_accuracies,
        label="Train Accuracy",
        linewidth=2,
    )

    plt.plot(
        epochs,
        validation_accuracies,
        label="Validation Accuracy",
        linewidth=2,
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "outputs/figures/accuracy_curve.png"
    )

    plt.close()

    print(
        "Training curves saved in outputs/figures/"
    )