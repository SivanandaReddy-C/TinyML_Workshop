"""
------------------------------------------------------------
Project SARA
Part VI - Deep Learning
Preparing the Dataset for PyTorch
------------------------------------------------------------
"""

import pandas as pd
import torch

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("../Datasets/dataset_processed.csv")

print("\nFirst Five Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())

print("\nDataset Shape:")
print(df.shape)

# ----------------------------------------------------------
# Separate Features and Labels
# ----------------------------------------------------------

print("\nSeparating Features and Labels...")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print("\nFeature Matrix Shape : ", X.shape)
print("Label Vector Shape   : ", y.shape)

# ----------------------------------------------------------
# Convert to PyTorch Tensors
# ----------------------------------------------------------

print("\nConverting to PyTorch Tensors...")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# ----------------------------------------------------------
# Verify Tensor Details
# ----------------------------------------------------------

print("\nTensor Information")

print("-------------------------------")
print("Feature Tensor")
print("-------------------------------")
print("Type :", type(X))
print("Shape:", X.shape)
print("Dtype:", X.dtype)

print("\n-------------------------------")
print("Label Tensor")
print("-------------------------------")
print("Type :", type(y))
print("Shape:", y.shape)
print("Dtype:", y.dtype)

# ----------------------------------------------------------
# Display One Sample
# ----------------------------------------------------------

print("\nFirst Training Sample")

print("Features:")
print(X[0])

print("\nLabel:")
print(y[0])

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("Dataset Preparation Completed Successfully")
print("=" * 60)

print(f"Feature Tensor : {X.shape}")
print(f"Label Tensor   : {y.shape}")

print("\nReady for the next step:")
print("Creating TensorDataset and DataLoader")