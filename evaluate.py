"""
evaluate.py
-----------
Evaluation script for FT-Transformer.

Author: Asal Javadpour
Project: XTab_Project
"""

import torch

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def evaluate(
    model,
    dataloader,
    device,
):
    """
    Evaluate model on the test dataset.
    """

    model.eval()

    predictions = []

    targets = []

    with torch.no_grad():

        for x_num, x_cat, labels in dataloader:

            x_num = x_num.to(device)

            x_cat = x_cat.to(device)

            outputs = model(
                x_numeric=x_num,
                x_categorical=x_cat,
            )

            predicted = torch.argmax(
                outputs,
                dim=1,
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

            targets.extend(
                labels.numpy()
            )

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    precision = precision_score(
        targets,
        predictions,
    )

    recall = recall_score(
        targets,
        predictions,
    )

    f1 = f1_score(
        targets,
        predictions,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }