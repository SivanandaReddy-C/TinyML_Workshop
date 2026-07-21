"""
------------------------------------------------------------
Project SARA
Part VI - Deep Learning
Export Trained MLP to ONNX
------------------------------------------------------------
"""

import os
import torch
import torch.nn as nn
import onnx

# ----------------------------------------------------------
# Create ONNX Directory
# ----------------------------------------------------------

os.makedirs("../ONNX", exist_ok=True)

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
# Load Trained Model
# ----------------------------------------------------------

print("=" * 60)
print("Loading Trained Model")
print("=" * 60)

model = ActivityClassifier()

model.load_state_dict(
    torch.load("../Models/mlp_activity_classifier.pth")
)

model.eval()

print("Model Loaded Successfully.")

# ----------------------------------------------------------
# Create Dummy Input
# ----------------------------------------------------------

dummy_input = torch.randn(1, 6)

print("\nDummy Input Shape :", dummy_input.shape)

# ----------------------------------------------------------
# Export to ONNX
# ----------------------------------------------------------

onnx_path = "../ONNX/mlp_activity_classifier.onnx"

print("\nExporting to ONNX...")

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    input_names=["input"],
    output_names=["output"],
    opset_version=17,
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    }
)

print("ONNX Model Exported Successfully.")

# ----------------------------------------------------------
# Validate ONNX Model
# ----------------------------------------------------------

print("\nValidating ONNX Model...")

onnx_model = onnx.load(onnx_path)

onnx.checker.check_model(onnx_model)

print("ONNX Model Validation Successful.")

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------

print("\n" + "=" * 60)
print("Generated File")
print("=" * 60)

print(onnx_path)