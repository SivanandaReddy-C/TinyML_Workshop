from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split

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

_, X_test, _, _ = train_test_split(
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
# Select One Sample
# -------------------------------------------------------

sample = X_test_tensor[0].unsqueeze(0)

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
# Warm-up
# -------------------------------------------------------

print("=" * 60)
print("PyTorch FP32 Profiling")
print("=" * 60)

print("Running Warm-up...")

with torch.no_grad():

    for _ in range(20):

        _ = model(sample)

print("Warm-up Completed")

# -------------------------------------------------------
# Profiling
# -------------------------------------------------------

NUM_RUNS = 1000

print(f"\nNumber of Profiling Runs : {NUM_RUNS}")

start = time.perf_counter()

with torch.no_grad():

    for _ in range(NUM_RUNS):

        outputs = model(sample)

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
print("PyTorch FP32 Performance")
print("=" * 60)

print(f"Device               : CPU")
print(f"Input Shape          : {tuple(sample.shape)}")
print(f"Number of Runs       : {NUM_RUNS}")
print(f"Average Latency      : {average_latency_ms:.6f} ms")
print(f"Throughput           : {throughput:.2f} inf/sec")

print("=" * 60)