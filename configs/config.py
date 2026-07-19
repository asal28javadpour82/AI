"""
config.py
---------
Central configuration file for the FT-Transformer project.

Author: Asal Javadpour
Project: XTab_Project
"""


class Config:

    # ==========================================
    # Random Seed
    # ==========================================

    RANDOM_SEED = 42

    # ==========================================
    # Dataset
    # ==========================================

    TEST_SIZE = 0.20

    VALIDATION_SIZE = 0.10

    NUM_CLASSES = 2

    NUM_NUMERIC_FEATURES = 6

    NUM_CATEGORICAL_FEATURES = 8

    # ==========================================
    # Training
    # ==========================================

    BATCH_SIZE = 256

    EPOCHS = 50

    LEARNING_RATE = 1e-3

    WEIGHT_DECAY = 1e-5

    # ==========================================
    # Model
    # ==========================================

    EMBEDDING_DIM = 32

    NUM_HEADS = 8

    NUM_TRANSFORMER_BLOCKS = 3

    FFN_MULTIPLIER = 4

    DROPOUT = 0.10

    # ==========================================
    # Early Stopping
    # ==========================================

    PATIENCE = 10

    # ==========================================
    # Device
    # ==========================================

    DEVICE = "cpu"

    # ==========================================
    # Checkpoints
    # ==========================================

    SAVE_MODEL_PATH = "outputs/checkpoints/best_model.pth"