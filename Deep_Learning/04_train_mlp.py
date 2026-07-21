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

# ----------------------------------------------------------
# Optimizer
# ----------------------------------------------------------

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

print("\n" + "=" * 60)
print("Optimizer")
print("=" * 60)

print(optimizer)


# ----------------------------------------------------------
# Training Configuration
# ----------------------------------------------------------

EPOCHS = 20

print("\n" + "=" * 60)
print("Training Started")
print("=" * 60)

# ----------------------------------------------------------
# Training Loop
# ----------------------------------------------------------

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for features, labels in loader:

        # Forward Pass
        outputs = model(features)

        # Compute Loss
        loss = criterion(outputs, labels)

        # Clear Previous Gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update Weights
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(loader)

    print(f"Epoch [{epoch+1:02d}/{EPOCHS}]   Loss = {epoch_loss:.6f}")

print("\n" + "=" * 60)
print("Training Completed Successfully")
print("=" * 60)