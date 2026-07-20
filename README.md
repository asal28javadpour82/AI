# XTab_Project

An implementation of **FT-Transformer** for tabular data classification using the **Adult Income Dataset** with an additional **Adaptive Feature Gating** module.

This project was developed as the final project for the **Special Topics (Machine Learning)** course.

---

# Project Overview

FT-Transformer is a Transformer-based deep learning architecture specifically designed for tabular datasets.

In this project, the original architecture has been implemented in PyTorch and extended by introducing an **Adaptive Feature Gating** mechanism before the Transformer encoder to improve feature representation.

---

# Features

- FT-Transformer implementation from scratch
- Multi-Head Self Attention
- Transformer Encoder Blocks
- Feature Tokenizer
- CLS Token
- Adaptive Feature Gating (Proposed Improvement)
- Early Stopping
- Learning Rate Scheduler
- Training & Validation Pipeline
- Automatic Evaluation
- Training Visualization
- Best Model Checkpoint Saving

---

# Project Structure

```
XTab_Project
│
├── configs/
│   └── config.py
│
├── data/
│   └── raw/
│       └── adult.data
│
├── models/
│   ├── cls_token.py
│   ├── feature_tokenizer.py
│   ├── feature_gating.py
│   ├── feedforward.py
│   ├── head.py
│   ├── multihead_attention.py
│   ├── transformer_block.py
│   └── transformer.py
│
├── outputs/
│   ├── checkpoints/
│   └── figures/
│
├── utils/
│   ├── dataloader.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── preprocessing.py
│   ├── torch_dataset.py
│   └── visualization.py
│
├── train.py
├── inference.py
├── requirements.txt
└── README.md
```

---

# Dataset

This project uses the **Adult Income Dataset** from the UCI Machine Learning Repository.

Prediction Task:

```
Income > 50K
or
Income <= 50K
```

---

# Model Architecture

```
Numerical Features
        │
Categorical Features
        │
        ▼
 Feature Tokenizer
        │
        ▼
 Adaptive Feature Gating
        │
        ▼
      CLS Token
        │
        ▼
 Transformer Blocks
        │
        ▼
    Layer Normalization
        │
        ▼
 Classification Head
        │
        ▼
 Income Prediction
```

---

# Training Configuration

| Parameter | Value |
|-----------|------:|
| Epochs | 50 |
| Batch Size | 256 |
| Embedding Dimension | 32 |
| Transformer Blocks | 3 |
| Attention Heads | 8 |
| Learning Rate | 1e-3 |
| Weight Decay | 1e-5 |
| Dropout | 0.10 |

---

# Final Results

| Metric | Score |
|--------|------:|
| Accuracy | **85.95%** |
| Precision | **72.50%** |
| Recall | **67.09%** |
| F1-Score | **69.69%** |

---

# Generated Outputs

The training process automatically generates:

- Best Model Checkpoint
- Loss Curve
- Accuracy Curve

These files are stored inside:

```
outputs/
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YourUsername/XTab_Project.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run Training

```bash
python train.py
```

---

# Run Inference

```bash
python inference.py
```

---

# Technologies

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

---

# Author

**Asal Javadpour**

Computer Engineering Student

Special Topics (Machine Learning) Course

2026

## License

This project was developed for educational purposes.