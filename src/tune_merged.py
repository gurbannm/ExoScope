import json, numpy as np, pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

INDIR = Path("data_input/processed_merged")
X = pd.read_csv(INDIR/"X_train.csv")
y = pd.read_csv(INDIR/"y_train.csv").squeeze("columns")

pipe = Pipeline([
    ("scaler", RobustScaler()),
    ("clf", HistGradientBoostingClassifier(loss="log_loss", random_state=42))
])

param_dist = {
    "clf__learning_rate":  np.logspace(-2.0, -0.8, 6),   # ~0.01..0.16
    "clf__max_depth":      [None, 3, 4, 5],
    "clf__l2_regularization": np.logspace(-4, -1, 5),    # 1e-4..1e-1
    "clf__max_iter":       [300, 500, 700],
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    pipe,
    param_distributions=param_dist,
    n_iter=12,                 # smaller search
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

search.fit(X, y)

Path("reports").mkdir(exist_ok=True)
out = {
    "best_params": search.best_params_,
    "best_score_macro_f1": float(search.best_score_)
}
Path("reports/hgb_tuning_merged.json").write_text(json.dumps(out, indent=2))
print("\n✅ Best macro-F1:", out["best_score_macro_f1"])
print("💾 Saved params to reports/hgb_tuning_merged.json")
