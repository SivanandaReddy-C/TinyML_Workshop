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