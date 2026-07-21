"""
------------------------------------------------------------
Project SARA
Part VI - Deep Learning
Model Evaluation
------------------------------------------------------------
"""

import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from torch.utils.data import TensorDataset, DataLoader

# ----------------------------------------------------------
# Define MLP
# ----------------------------------------------------------

class ActivityClassifier(nn.Module):

    def __init__(self):

        super(ActivityClassifier, self).__init__()

        self.network = nn.Sequential(

            nn.Linear(6,32),
            nn.ReLU(),

            nn.Linear(32,16),
            nn.ReLU(),

            nn.Linear(16,5)

        )

    def forward(self,x):

        return self.network(x)

# ----------------------------------------------------------
# Load Dataset
# ----------------------------------------------------------

df = pd.read_csv("../Datasets/dataset_processed.csv")

X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

X = torch.tensor(X,dtype=torch.float32)
y = torch.tensor(y,dtype=torch.long)

dataset = TensorDataset(X,y)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=False
)

# ----------------------------------------------------------
# Load Trained Model
# ----------------------------------------------------------

model = ActivityClassifier()

model.load_state_dict(
    torch.load("../Models/mlp_activity_classifier.pth")
)

model.eval()

# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

predictions = []
ground_truth = []

with torch.no_grad():

    for features,labels in loader:

        outputs = model(features)

        predicted = torch.argmax(outputs,dim=1)

        predictions.extend(predicted.numpy())

        ground_truth.extend(labels.numpy())

# ----------------------------------------------------------
# Metrics
# ----------------------------------------------------------

accuracy = accuracy_score(
    ground_truth,
    predictions
)

print("="*60)

print("Model Accuracy")

print("="*60)

print(f"Accuracy : {accuracy*100:.2f}%")

print("\nConfusion Matrix")

print(confusion_matrix(
    ground_truth,
    predictions
))

print("\nClassification Report")

print(classification_report(
    ground_truth,
    predictions
))