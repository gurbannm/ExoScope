import argparse, joblib, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

def run(model_path: Path, processed_dir: Path, out_csv: Path):
    bundle = joblib.load(model_path)
    clf, scaler = bundle["model"], bundle["scaler"]
    feats, classes = bundle["features"], bundle["classes"]

    X_test = pd.read_csv(processed_dir / "X_test.csv")[feats]
    y_test = pd.read_csv(processed_dir / "y_test.csv").squeeze("columns")
    X_test_s = scaler.transform(X_test)

    r = permutation_importance(clf, X_test_s, y_test, n_repeats=8, random_state=42, scoring="f1_macro")
    imp = pd.Series(r.importances_mean, index=feats).sort_values(ascending=False)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    imp.to_csv(out_csv, header=["importance"])
    print(f"✅ wrote feature importance to {out_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/planet_classifier.joblib")
    parser.add_argument("--data",  default="data_input/processed")
    parser.add_argument("--out",   default="reports/top_features.csv")
    args = parser.parse_args()
    run(Path(args.model), Path(args.data), Path(args.out))
