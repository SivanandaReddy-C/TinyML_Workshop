from pathlib import Path
import time
import numpy as np
import pandas as pd
import onnxruntime as ort

from sklearn.model_selection import train_test_split

# -------------------------------------------------------
# Random Seed
# -------------------------------------------------------

SEED = 42

np.random.seed(SEED)

# -------------------------------------------------------
# Project Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_PATH = PROJECT_ROOT / "Datasets" / "dataset_processed.csv"

MODEL_PATH = PROJECT_ROOT / "ONNX" / "mlp_model.onnx"

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

df = pd.read_csv(DATASET_PATH)

X = df.drop(columns=["Label"]).values

y = df["Label"].values

# -------------------------------------------------------
# Frozen Train-Test Split
# -------------------------------------------------------

_, X_test, _, _ = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y,
)

# -------------------------------------------------------
# Select One Sample
# -------------------------------------------------------

sample = X_test[0].astype(np.float32).reshape(1, 6)

# -------------------------------------------------------
# Load ONNX Model
# -------------------------------------------------------

session = ort.InferenceSession(
    str(MODEL_PATH),
    providers=["CPUExecutionProvider"],
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# -------------------------------------------------------
# Warm-up
# -------------------------------------------------------

print("=" * 60)
print("ONNX FP32 Profiling")
print("=" * 60)

print("Running Warm-up...")

for _ in range(20):

    session.run(
        [output_name],
        {input_name: sample},
    )

print("Warm-up Completed")

# -------------------------------------------------------
# Profiling
# -------------------------------------------------------

NUM_RUNS = 1000

print(f"\nNumber of Profiling Runs : {NUM_RUNS}")

start = time.perf_counter()

for _ in range(NUM_RUNS):

    outputs = session.run(
        [output_name],
        {input_name: sample},
    )

end = time.perf_counter()

# -------------------------------------------------------
# Performance Metrics
# -------------------------------------------------------

total_time = end - start

average_latency_ms = (total_time / NUM_RUNS) * 1000

throughput = NUM_RUNS / total_time

# -------------------------------------------------------
# Display Results
# -------------------------------------------------------

print("\n" + "=" * 60)
print("ONNX FP32 Performance")
print("=" * 60)

print(f"Execution Provider   : {session.get_providers()[0]}")
print(f"Input Shape          : {sample.shape}")
print(f"Number of Runs       : {NUM_RUNS}")
print(f"Average Latency      : {average_latency_ms:.6f} ms")
print(f"Throughput           : {throughput:.2f} inf/sec")

print("=" * 60)