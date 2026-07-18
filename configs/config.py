"""
Project Configuration File
--------------------------
XTab-Lite Project
AI Laboratory Project
"""

# ===========================
# Dataset
# ===========================

DATASET_NAME = "Adult Income"

TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42

# ===========================
# Training
# ===========================

BATCH_SIZE = 64

NUM_EPOCHS = 50

LEARNING_RATE = 1e-3

WEIGHT_DECAY = 1e-5

# ===========================
# Model
# ===========================

EMBEDDING_DIM = 64

NUM_HEADS = 8

NUM_TRANSFORMER_LAYERS = 4

FFN_DIM = 256

DROPOUT = 0.2

# ===========================
# Device
# ===========================

DEVICE = "cuda"

# ===========================
# Save Paths
# ===========================

CHECKPOINT_PATH = "outputs/checkpoints"

FIGURE_PATH = "outputs/figures"

LOG_PATH = "outputs/logs"

# ===========================
# OpenML Dataset
# ===========================

OPENML_DATASET_NAME = "adult"

TARGET_COLUMN = "class"