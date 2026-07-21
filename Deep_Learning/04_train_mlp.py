"""
------------------------------------------------------------
Project SARA
Part VI - Deep Learning
Loss Function
------------------------------------------------------------
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ----------------------------------------------------------
# Define MLP
# ----------------------------------------------------------

class ActivityClassifier(nn.Module):

    def __init__(self):

        super(ActivityClassifier, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(6, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 5)

        )

    def forward(self, x):

        return self.network(x)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = pd.read_csv("../Datasets/dataset_processed.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# ----------------------------------------------------------
# Create Model
# ----------------------------------------------------------

model = ActivityClassifier()

# ----------------------------------------------------------
# Loss Function
# ----------------------------------------------------------

criterion = nn.CrossEntropyLoss()

print("=" * 60)
print("Loss Function")
print("=" * 60)

print(criterion)

# ----------------------------------------------------------
# One Mini Batch
# ----------------------------------------------------------

features, labels = next(iter(loader))

outputs = model(features)

# ----------------------------------------------------------
# Compute Loss
# ----------------------------------------------------------

loss = criterion(outputs, labels)

print("\nLoss Value")

print(loss)

print("\nLoss (float)")

print(loss.item())