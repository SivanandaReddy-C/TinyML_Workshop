import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load dataset
df = pd.read_csv("../Datasets/dataset.csv")

# Normalize IMU features
features = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Encode labels
encoder = LabelEncoder()
df["Label"] = encoder.fit_transform(df["Label"])

# Save processed dataset
df.to_csv("../Datasets/dataset_processed.csv", index=False)

print("Processed dataset saved successfully!")