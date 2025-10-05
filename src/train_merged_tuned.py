import json
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier

# --- paths ---
INDIR  = Path("data_input/processed_merged")
OUTDIR = Path("models"); OUTDIR.mkdir(exist_ok=True)
REPORTS = Path("reports")

# --- load splits ---
X_train = pd.read_csv(INDIR/"X_train.csv")
X_val   = pd.read_csv(INDIR/"X_val.csv")
X_test  = pd.read_csv(INDIR/"X_test.csv")
y_train = pd.read_csv(INDIR/"y_train.csv").squeeze("columns")
y_val   = pd.read_csv(INDIR/"y_val.csv").squeeze("columns")
y_test  = pd.read_csv(INDIR/"y_test.csv").squeeze("columns")

# --- scale ---
scaler = RobustScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s   = scaler.transform(X_val)
X_test_s  = scaler.transform(X_test)

# --- class weights ---
classes = np.unique(y_train)
weights = compute_class_weight("balanced", classes=classes, y=y_train)
class_weight = {c:w for c,w in zip(classes, weights)}
sample_weight = np.array([class_weight[c] for c in y_train])

# --- read tuned params (fallback to sensible defaults) ---
best_params_path = REPORTS/"hgb_tuning_merged.json"
base_params = dict(
    loss="log_loss",
    learning_rate=0.06,
    max_iter=600,
    max_depth=4,
    l2_regularization=1e-3,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20
)
if best_params_path.exists():
    tuned = json.loads(best_params_path.read_text())["best_params"]
    # flatten keys like "clf__max_depth" -> "max_depth"
    tuned = {k.split("__",1)[1]: v for k,v in tuned.items()}
    base_params.update(tuned)
    print("🔧 Using tuned params:", base_params)
else:
    print("ℹ️ No tuned params file found; using defaults:", base_params)

# --- train ---
clf = HistGradientBoostingClassifier(**base_params)
clf.fit(X_train_s, y_train, sample_weight=sample_weight)

# --- evaluate ---
y_pred = clf.predict(X_test_s)
rep = classification_report(y_test, y_pred, digits=3, output_dict=True)
cm  = confusion_matrix(y_test, y_pred, labels=classes).tolist()
print("✅ Test classification report (merged tuned):")
import pprint; pprint.pprint(rep)
print("Confusion matrix:", cm)

# --- save model + metrics ---
joblib.dump(
    {"model": clf, "scaler": scaler, "features": X_train.columns.tolist(), "classes": list(classes)},
    OUTDIR/"planet_classifier_merged_tuned.joblib"
)
REPORTS.mkdir(exist_ok=True)
with open(REPORTS/"metrics_merged_tuned.json","w") as f:
    json.dump({"report": rep, "confusion_matrix": cm}, f, indent=2)

print("💾 Saved: models/planet_classifier_merged_tuned.joblib")
