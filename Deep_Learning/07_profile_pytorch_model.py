"""
------------------------------------------------------------
Project SARA
Part VI - Deep Learning
Profile Trained PyTorch Model
------------------------------------------------------------
"""

import os
import torch
import torch.nn as nn

# ==========================================================
# Define MLP Architecture
# ==========================================================

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


# ==========================================================
# Load Trained Model
# ==========================================================

MODEL_PATH = "../Models/mlp_activity_classifier.pth"

model = ActivityClassifier()

model.load_state_dict(
    torch.load(MODEL_PATH)
)

model.eval()

print("=" * 70)
print("TRAINED MODEL LOADED SUCCESSFULLY")
print("=" * 70)


# ==========================================================
# Display Model Architecture
# ==========================================================

print("\nMODEL ARCHITECTURE")
print("-" * 70)
print(model)


# ==========================================================
# Count Parameters
# ==========================================================

total_parameters = sum(
    p.numel() for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\n" + "=" * 70)
print("MODEL PARAMETERS")
print("=" * 70)

print(f"Total Parameters      : {total_parameters:,}")
print(f"Trainable Parameters  : {trainable_parameters:,}")


# ==========================================================
# Compute Raw Parameter Memory
# ==========================================================

BYTES_PER_PARAMETER = 4          # FP32

raw_parameter_memory = (
    total_parameters *
    BYTES_PER_PARAMETER
)

raw_parameter_memory_kb = (
    raw_parameter_memory / 1024
)


# ==========================================================
# Compute Serialized Model Size
# ==========================================================

model_size_bytes = os.path.getsize(MODEL_PATH)

model_size_kb = model_size_bytes / 1024

serialization_overhead = (
    model_size_bytes -
    raw_parameter_memory
)

serialization_overhead_kb = (
    serialization_overhead / 1024
)


print("\n" + "=" * 70)
print("MODEL MEMORY PROFILE")
print("=" * 70)

print(f"Parameter Precision        : FP32")
print(f"Bytes per Parameter        : {BYTES_PER_PARAMETER}")

print()

print(f"Raw Parameter Memory       : {raw_parameter_memory} Bytes")
print(f"                           : {raw_parameter_memory_kb:.2f} KB")

print()

print(f"Serialized Model (.pth)    : {model_size_bytes} Bytes")
print(f"                           : {model_size_kb:.2f} KB")

print()

print(f"Serialization Overhead     : {serialization_overhead} Bytes")
print(f"                           : {serialization_overhead_kb:.2f} KB")


# ==========================================================
# Layer-wise Summary
# ==========================================================

print("\n" + "=" * 70)
print("LAYER-WISE SUMMARY")
print("=" * 70)

for name, layer in model.named_modules():

    if isinstance(layer, nn.Linear):

        print()

        print(name)

        print(f"Input Features   : {layer.in_features}")

        print(f"Output Features  : {layer.out_features}")

        print(f"Weight Shape     : {tuple(layer.weight.shape)}")

        print(f"Bias Shape       : {tuple(layer.bias.shape)}")

        weights = layer.weight.numel()

        bias = layer.bias.numel()

        total = weights + bias

        print(f"Weights          : {weights}")

        print(f"Biases           : {bias}")

        print(f"Layer Parameters : {total}")


# ==========================================================
# Save Report
# ==========================================================

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

REPORT_PATH = os.path.join(
    OUTPUT_DIR,
    "pytorch_profile.txt"
)

with open(REPORT_PATH, "w") as f:

    f.write("PyTorch Model Profile\n")
    f.write("=" * 70 + "\n\n")

    f.write("MODEL ARCHITECTURE\n")
    f.write("-" * 70 + "\n")
    f.write(str(model))
    f.write("\n\n")

    f.write("MODEL PARAMETERS\n")
    f.write("-" * 70 + "\n")

    f.write(f"Total Parameters      : {total_parameters}\n")
    f.write(f"Trainable Parameters  : {trainable_parameters}\n\n")

    f.write("MODEL MEMORY PROFILE\n")
    f.write("-" * 70 + "\n")

    f.write(f"Parameter Precision        : FP32\n")
    f.write(f"Bytes per Parameter        : {BYTES_PER_PARAMETER}\n\n")

    f.write(
        f"Raw Parameter Memory       : "
        f"{raw_parameter_memory} Bytes "
        f"({raw_parameter_memory_kb:.2f} KB)\n"
    )

    f.write(
        f"Serialized Model (.pth)    : "
        f"{model_size_bytes} Bytes "
        f"({model_size_kb:.2f} KB)\n"
    )

    f.write(
        f"Serialization Overhead     : "
        f"{serialization_overhead} Bytes "
        f"({serialization_overhead_kb:.2f} KB)\n\n"
    )

    f.write("LAYER-WISE SUMMARY\n")
    f.write("-" * 70 + "\n")

    for name, layer in model.named_modules():

        if isinstance(layer, nn.Linear):

            weights = layer.weight.numel()

            bias = layer.bias.numel()

            total = weights + bias

            f.write(f"\n{name}\n")

            f.write(
                f"Input Features   : {layer.in_features}\n"
            )

            f.write(
                f"Output Features  : {layer.out_features}\n"
            )

            f.write(
                f"Weight Shape     : "
                f"{tuple(layer.weight.shape)}\n"
            )

            f.write(
                f"Bias Shape       : "
                f"{tuple(layer.bias.shape)}\n"
            )

            f.write(f"Weights          : {weights}\n")

            f.write(f"Biases           : {bias}\n")

            f.write(f"Layer Parameters : {total}\n")


# ==========================================================
# Final Summary
# ==========================================================

print("\n" + "=" * 70)
print("PROFILE COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"Report Saved To : {REPORT_PATH}")