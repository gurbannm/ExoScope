import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

IN = Path("data_input/merged/merged_raw.csv")
OUT = Path("data_input/processed_merged"); OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(IN)

# numeric feature columns (ignore label + mission)
feature_cols = [c for c in df.columns if c not in {"label","mission"} and pd.api.types.is_numeric_dtype(df[c])]
X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
y = df["label"]

# split data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

# save processed splits
X_train.to_csv(OUT / "X_train.csv", index=False)
X_val.to_csv(OUT / "X_val.csv", index=False)
X_test.to_csv(OUT / "X_test.csv", index=False)
y_train.to_csv(OUT / "y_train.csv", index=False)
y_val.to_csv(OUT / "y_val.csv", index=False)
y_test.to_csv(OUT / "y_test.csv", index=False)

print("✅ processed merged dataset saved to", OUT)
