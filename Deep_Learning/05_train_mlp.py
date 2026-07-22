from pathlib import Path
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from build_mlp import MLP

# -------------------------------------------------------
# Fix Random Seeds
# -------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

dataset_path = PROJECT_ROOT / "Datasets" / "dataset_processed.csv"

df = pd.read_csv(dataset_path)

# -------------------------------------------------------
# Features and Labels
# -------------------------------------------------------

X = df.drop(columns=["Label"]).values

y = df["Label"].values

# -------------------------------------------------------
# Train-Test Split
# -------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

# -------------------------------------------------------
# Convert to PyTorch Tensors
# -------------------------------------------------------

X_train = torch.tensor(X_train, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)

# -------------------------------------------------------
# Dataset and DataLoader
# -------------------------------------------------------

train_dataset = TensorDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

# -------------------------------------------------------
# Model
# -------------------------------------------------------

model = MLP()

# -------------------------------------------------------
# Loss Function
# -------------------------------------------------------

criterion = nn.CrossEntropyLoss()

# -------------------------------------------------------
# Optimizer
# -------------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("=" * 60)
print("Training Pipeline Configured Successfully")
print("=" * 60)

print(f"\nTraining Samples : {len(train_dataset)}")
print(f"Mini-batch Size  : 32")
print(f"Total Batches    : {len(train_loader)}")

print("\nLoss Function")
print(criterion)

print("\nOptimizer Configuration")
print(f"Optimizer      : Adam")
print(f"Learning Rate  : {optimizer.param_groups[0]['lr']}")
print(f"Batch Size     : {train_loader.batch_size}")
print(f"Epochs         : 20")

print("\nTraining Configuration")
print("-" * 40)
print(f"Input Features     : 6")
print(f"Output Classes     : 5")
print(f"Hidden Layers      : 64 -> 32")
print(f"Training Samples   : {len(train_dataset)}")
print(f"Batch Size         : 32")
print(f"Epochs             : 20")
print(f"Learning Rate      : 0.001")
print(f"Loss Function      : CrossEntropyLoss")
print(f"Optimizer          : Adam")