"""
------------------------------------------------------------
Project SARA
Part VI - Deep Learning
Building the Multi-Layer Perceptron (MLP)
------------------------------------------------------------
"""

import torch
import torch.nn as nn

# ----------------------------------------------------------
# Define the MLP Architecture
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
# Create Model
# ----------------------------------------------------------

model = ActivityClassifier()

# ----------------------------------------------------------
# Display Model
# ----------------------------------------------------------

print("=" * 60)
print("Project SARA - Multi-Layer Perceptron")
print("=" * 60)

print(model)

# ----------------------------------------------------------
# Display Model Parameters
# ----------------------------------------------------------

print("\nModel Parameters\n")

total_params = 0

for name, param in model.named_parameters():

    print(f"{name:30} {list(param.shape)}")

    total_params += param.numel()

print("\nTotal Trainable Parameters :", total_params)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_csv("../Datasets/dataset_processed.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X, y)

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# ----------------------------------------------------------
# Get One Mini-Batch
# ----------------------------------------------------------

features, labels = next(iter(train_loader))

print("\nMini-Batch Shape")
print("-----------------------------")
print("Features :", features.shape)
print("Labels   :", labels.shape)

# ----------------------------------------------------------
# Forward Propagation
# ----------------------------------------------------------

outputs = model(features)

print("\nForward Propagation Completed")

print("\nOutput Shape")
print(outputs.shape)

print("\nRaw Network Outputs (First Sample)")
print(outputs[0])

# ----------------------------------------------------------
# Predicted Class
# ----------------------------------------------------------

predictions = torch.argmax(outputs, dim=1)

print("\nPredicted Classes")
print(predictions)

print("\nGround Truth Labels")
print(labels)