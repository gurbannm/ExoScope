import pandas as pd
import numpy as np
from pandas.errors import ParserError
from sklearn.model_selection import train_test_split
from pathlib import Path

# ---- paths (relative is simpler and portable) ----
RAW = Path("data_input/tess_toi.csv")          # your dataset file
OUTDIR = Path("data_input/processed")          # where outputs go
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---- robust load: auto-detect delimiter, skip comments/malformed lines ----
try:
    df = pd.read_csv(RAW, sep=None, engine="python", comment="#", on_bad_lines="skip")
except ParserError:
    # common fallback for NASA tables
    try:
        df = pd.read_csv(RAW, sep="\t", engine="python", comment="#", on_bad_lines="skip")
    except ParserError:
        df = pd.read_csv(RAW, sep=";", engine="python", comment="#", on_bad_lines="skip")

print("✅ Data loaded:", df.shape)

# ---- normalize column names once ----
orig_cols = df.columns.tolist()
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

# ---- find disposition column on lowercased names ----
candidates = [
    "tfopwg_disposition", "tfop_disposition", "archive_disposition",
    "disposition", "toi_disposition", "tfopwg_disp", "disp"
]
disp_col = next((c for c in candidates if c in df.columns), None)
if disp_col is None:
    # last resort: anything containing 'disposition'
    poss = [c for c in df.columns if "disposition" in c]
    disp_col = poss[0] if poss else None

if disp_col is None:
    print("⚠️ No disposition-like column found. Columns are:\n", orig_cols)
    raise RuntimeError("Re-download the CSV including 'TFOPWG Disposition'.")

# ---- labels: Planet / Candidate / False Positive ----
def map_label(x: str) -> str:
    s = str(x).upper()
    if "CONFIRMED" in s or s in {"CP", "CONFIRM"}:
        return "Planet"
    if "FALSE" in s or s in {"FP", "FALSE_POSITIVE"}:
        return "False Positive"
    # PC, KP, APC, or anything else → Candidate
    return "Candidate"

df["label"] = df[disp_col].map(map_label)

# ---- choose numeric features safely ----
num_all = df.select_dtypes(include=[np.number]).columns.tolist()

# drop obvious IDs / indices
drop_like = {"tic", "toi", "id", "kepid", "index"}
num_all = [c for c in num_all if not any(k in c for k in drop_like)]

# drop columns that are all-NaN
num_all = [c for c in num_all if not df[c].isna().all()]

if not num_all:
    raise RuntimeError("No usable numeric features found.")

# clean values
X = df[num_all].copy()
X = X.replace([np.inf, -np.inf], np.nan)

# column-wise median fill; for columns that remain all-NaN, fill with 0
col_medians = X.median(numeric_only=True)
X = X.fillna(col_medians).fillna(0.0)

y = df["label"]

# ---- split ----
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

# ---- save ----
X_train.to_csv(OUTDIR / "X_train.csv", index=False)
X_val.to_csv(OUTDIR / "X_val.csv", index=False)
X_test.to_csv(OUTDIR / "X_test.csv", index=False)
y_train.to_csv(OUTDIR / "y_train.csv", index=False)
y_val.to_csv(OUTDIR / "y_val.csv", index=False)
y_test.to_csv(OUTDIR / "y_test.csv", index=False)

print("✅ Preprocessing complete!")
features = X.columns.tolist()
print("Features used:", len(features), "→", ", ".join(features[:12]) + ("..." if len(features) > 12 else ""))
print("Saved to 'data_input/processed/'")
