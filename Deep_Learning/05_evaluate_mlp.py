"""
------------------------------------------------------------
Project SARA
Part VI - Deep Learning
Model Evaluation
------------------------------------------------------------
"""

import os
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from torch.utils.data import TensorDataset, DataLoader

# ----------------------------------------------------------
# Create Output Directory
# ----------------------------------------------------------

os.makedirs("outputs", exist_ok=True)

# ----------------------------------------------------------
# Define MLP Architecture
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

print("=" * 60)
print("Loading Dataset...")
print("=" * 60)

df = pd.read_csv("../Datasets/dataset_processed.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X, y)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
)

print(f"Total Samples : {len(dataset)}")

# ----------------------------------------------------------
# Load Trained Model
# ----------------------------------------------------------

print("\nLoading Trained Model...")

model = ActivityClassifier()

model.load_state_dict(
    torch.load("../Models/mlp_activity_classifier.pth")
)

model.eval()

print("Model Loaded Successfully.")

# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

predictions = []
ground_truth = []

print("\nRunning Inference...")

with torch.no_grad():

    for features, labels in loader:

        outputs = model(features)

        predicted = torch.argmax(outputs, dim=1)

        predictions.extend(predicted.numpy())

        ground_truth.extend(labels.numpy())

# ----------------------------------------------------------
# Compute Metrics
# ----------------------------------------------------------

accuracy = accuracy_score(
    ground_truth,
    predictions
)

cm = confusion_matrix(
    ground_truth,
    predictions
)

report = classification_report(
    ground_truth,
    predictions
)

# ----------------------------------------------------------
# Display Results
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("Model Evaluation")
print("=" * 60)

print(f"\nAccuracy : {accuracy * 100:.2f}%")

print("\nConfusion Matrix")

print(cm)

print("\nClassification Report")

print(report)

# ----------------------------------------------------------
# Save Classification Report
# ----------------------------------------------------------

with open(
    "outputs/classification_report.txt",
    "w"
) as f:

    f.write("Model Accuracy\n")
    f.write("-------------------------\n")
    f.write(f"{accuracy*100:.2f}%\n\n")

    f.write("Confusion Matrix\n")
    f.write(str(cm))

    f.write("\n\n")

    f.write(report)

# ----------------------------------------------------------
# Save Evaluation Summary
# ----------------------------------------------------------

with open(
    "outputs/evaluation_summary.txt",
    "w"
) as f:

    f.write("Project SARA\n")
    f.write("Deep Learning Model Evaluation\n\n")

    f.write(f"Accuracy : {accuracy*100:.2f}%\n")

    f.write(f"Total Samples : {len(dataset)}\n")

    f.write("Model : MLP (6 -> 32 -> 16 -> 5)\n")

# ----------------------------------------------------------
# Plot Confusion Matrix
# ----------------------------------------------------------

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Idle",
        "Pitch",
        "Roll",
        "Yaw",
        "Shake"
    ]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    "outputs/confusion_matrix.png",
    dpi=300
)

plt.show()

# ----------------------------------------------------------
# Final Summary
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("Evaluation Completed Successfully")
print("=" * 60)

print("\nGenerated Files")

print("-------------------------------")

print("outputs/confusion_matrix.png")

print("outputs/classification_report.txt")

print("outputs/evaluation_summary.txt")