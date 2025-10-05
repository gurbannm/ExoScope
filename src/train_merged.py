import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier

INDIR = Path("data_input/processed_merged")
OUTDIR = Path("models"); OUTDIR.mkdir(exist_ok=True)

# load processed splits
X_train = pd.read_csv(INDIR / "X_train.csv")
X_val   = pd.read_csv(INDIR / "X_val.csv")
X_test  = pd.read_csv(INDIR / "X_test.csv")
y_train = pd.read_csv(INDIR / "y_train.csv").squeeze("columns")
y_val   = pd.read_csv(INDIR / "y_val.csv").squeeze("columns")
y_test  = pd.read_csv(INDIR / "y_test.csv").squeeze("columns")

# scale
scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# class weights for imbalance
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight = {cls: w for cls, w in zip(classes, weights)}
sample_weight = np.array([class_weight[c] for c in y_train])

# model
clf = HistGradientBoostingClassifier(
    loss="log_loss",
    learning_rate=0.06,
    max_iter=600,
    max_depth=None,
    l2_regularization=0.0,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20
)

clf.fit(X_train_s, y_train, sample_weight=sample_weight)

# evaluation
y_pred = clf.predict(X_test_s)
rep = classification_report(y_test, y_pred, digits=3, output_dict=True)
cm  = confusion_matrix(y_test, y_pred, labels=classes).tolist()

print("✅ Test classification report (merged):")
print(json.dumps(rep, indent=2))
print("Confusion matrix:", cm)

# save model bundle
joblib.dump(
    {"model": clf, "scaler": scaler, "features": X_train.columns.tolist(), "classes": list(classes)},
    OUTDIR / "planet_classifier_merged.joblib"
)
Path("reports").mkdir(exist_ok=True)
with open("reports/metrics_merged.json", "w") as f:
    json.dump({"report": rep, "confusion_matrix": cm}, f, indent=2)

print("💾 Saved: models/planet_classifier_merged.joblib")
