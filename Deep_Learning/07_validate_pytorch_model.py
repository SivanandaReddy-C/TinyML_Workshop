from pathlib import Path
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from build_mlp import MLP

# -------------------------------------------------------
# Random Seed
# -------------------------------------------------------

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -------------------------------------------------------
# Project Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "Datasets" / "dataset_processed.csv"

MODEL_PATH = PROJECT_ROOT / "Models" / "mlp_model.pth"

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

df = pd.read_csv(DATASET_PATH)

X = df.drop(columns=["Label"]).values

y = df["Label"].values

# -------------------------------------------------------
# Frozen Train-Test Split
# -------------------------------------------------------

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)

# -------------------------------------------------------
# Convert to Tensor
# -------------------------------------------------------

X_test_tensor = torch.tensor(
    X_test,
    dtype=torch.float32,
)

# -------------------------------------------------------
# Load Model
# -------------------------------------------------------

model = MLP()

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=torch.device("cpu"),
    )
)

model.eval()

# -------------------------------------------------------
# Inference
# -------------------------------------------------------

with torch.no_grad():

    outputs = model(X_test_tensor)

    predictions = torch.argmax(
        outputs,
        dim=1,
    ).numpy()

# -------------------------------------------------------
# Metrics
# -------------------------------------------------------

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    average="weighted",
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted",
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
)

cm = confusion_matrix(
    y_test,
    predictions,
)

print("=" * 60)
print("PyTorch FP32 Validation")
print("=" * 60)

print(f"Accuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1-Score  : {f1*100:.2f}%")

print("\nConfusion Matrix")

print(cm)

print("\nClassification Report")

print(classification_report(y_test, predictions))