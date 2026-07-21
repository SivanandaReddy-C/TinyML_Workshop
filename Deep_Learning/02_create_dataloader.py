"""
------------------------------------------------------------
Project SARA
Part VI - Deep Learning
Slide 16 : Creating TensorDataset and DataLoader
------------------------------------------------------------
"""

import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("../Datasets/dataset_processed.csv")

# ----------------------------------------------------------
# Separate Features and Labels
# ----------------------------------------------------------

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# ----------------------------------------------------------
# Convert to PyTorch Tensors
# ----------------------------------------------------------

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# ----------------------------------------------------------
# Create TensorDataset
# ----------------------------------------------------------

dataset = TensorDataset(X, y)

print("\nTensorDataset Created Successfully")
print(f"Number of Samples : {len(dataset)}")

# ----------------------------------------------------------
# Create DataLoader
# ----------------------------------------------------------

BATCH_SIZE = 32

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print("\nDataLoader Created Successfully")
print(f"Batch Size : {BATCH_SIZE}")
print(f"Number of Mini-Batches : {len(train_loader)}")

# ----------------------------------------------------------
# Inspect One Mini-Batch
# ----------------------------------------------------------

print("\nFetching First Mini-Batch...\n")

features, labels = next(iter(train_loader))

print("Feature Batch Shape :", features.shape)
print("Label Batch Shape   :", labels.shape)

print("\nFeature Tensor Type :", features.dtype)
print("Label Tensor Type   :", labels.dtype)

# ----------------------------------------------------------
# Display First Mini-Batch
# ----------------------------------------------------------

print("\nFirst Mini-Batch (First Five Samples)")
print("-" * 60)

for i in range(5):
    print(f"\nSample {i+1}")
    print("Features :", features[i])
    print("Label    :", labels[i])

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("DataLoader Ready for Neural Network Training")
print("=" * 60)

print(f"Total Samples : {len(dataset)}")
print(f"Batch Size    : {BATCH_SIZE}")
print(f"Mini-Batches  : {len(train_loader)}")