from pathlib import Path
import random
import numpy as np
import pandas as pd
import onnxruntime as ort

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# -------------------------------------------------------
# Random Seed
# -------------------------------------------------------

SEED = 42

random.seed(SEED)
np.random.seed(SEED)

# -------------------------------------------------------
# Project Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "Datasets" / "dataset_processed.csv"

ONNX_PATH = PROJECT_ROOT / "ONNX" / "mlp_model.onnx"

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

df = pd.read_csv(DATASET_PATH)

X = df.drop(columns=["Label"]).values.astype(np.float32)

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
# Load ONNX Model
# -------------------------------------------------------

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name

output_name = session.get_outputs()[0].name

# -------------------------------------------------------
# Run Inference
# -------------------------------------------------------

predictions = []

for sample in X_test:
    sample = sample.reshape(1, 6).astype(np.float32)

    output = session.run(
        [output_name],
        {input_name: sample}
    )[0]

    predictions.append(np.argmax(output))

# -------------------------------------------------------
# Performance Metrics
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
print("ONNX FP32 Validation")
print("=" * 60)

print(f"\nAccuracy  : {accuracy*100:.2f}%")
print(f"Precision : {precision*100:.2f}%")
print(f"Recall    : {recall*100:.2f}%")
print(f"F1-Score  : {f1*100:.2f}%")

print("\nConfusion Matrix")
print(cm)

print("\nClassification Report")
print(classification_report(y_test, predictions))