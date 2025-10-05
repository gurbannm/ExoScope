import sys
import pandas as pd
import joblib
from pathlib import Path

# Load trained model bundle
bundle = joblib.load("models/planet_classifier.joblib")
clf = bundle["model"]
scaler = bundle["scaler"]
features = bundle["features"]
classes = bundle["classes"]

# Default: use first row of X_test
if len(sys.argv) > 1:
    row_file = Path(sys.argv[1])
else:
    row_file = Path("data_input/processed/X_test.csv")

X = pd.read_csv(row_file)[features]
x = X.iloc[[0]]  # first row
x_s = scaler.transform(x)

proba = clf.predict_proba(x_s)[0]
pred = classes[proba.argmax()]

print("🔮 Predicted:", pred)
print("📊 Probabilities:")
for cls, p in zip(classes, proba):
    print(f"  {cls}: {p:.3f}")
