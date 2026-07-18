import joblib
import numpy as np

model = joblib.load("Models/random_forest_model.pkl")

x = np.array([[0.506, 0.509, 0.753, 0.493, 0.499, 0.500]])

print("Prediction:", model.predict(x))
print("Probabilities:", model.predict_proba(x))

