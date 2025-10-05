import pandas as pd
import numpy as np
from pathlib import Path
from pandas.errors import ParserError

DATA = Path("data_input")
OUT = DATA / "merged"
OUT.mkdir(parents=True, exist_ok=True)

FILES = [
    ("tess", DATA/"tess_toi.csv",            "tfopwg_disposition"),
    ("k2",   DATA/"k2_pandc.csv",            "archive_disposition"),
    ("all",  DATA/"cumulative_exoplanets.csv", "disposition"),  # adjust if diff
]

def robust_read(p):
    try:
        return pd.read_csv(p, sep=None, engine="python", comment="#", on_bad_lines="skip")
    except ParserError:
        for sep in ["\t",";"]:
            try: return pd.read_csv(p, sep=sep, engine="python", comment="#", on_bad_lines="skip")
            except ParserError: pass
    raise RuntimeError(f"Could not parse {p}")

def norm_cols(df):
    df = df.copy()
    df.columns = [c.strip().lower().replace(" ","_") for c in df.columns]
    return df

def map_label(x: str) -> str:
    s = str(x).upper()
    if "CONFIRMED" in s or s in {"CP","CONFIRM"}: return "Planet"
    if "FALSE" in s or s in {"FP","FALSE_POSITIVE"}: return "False Positive"
    return "Candidate"

frames = []
for mission, path, disp_hint in FILES:
    if not path.exists(): 
        print(f"⚠️ skip {mission}: {path.name} not found")
        continue
    df = robust_read(path)
    df = norm_cols(df)
    # find a disposition-like column
    candidates = [disp_hint, "tfopwg_disposition","archive_disposition","disposition","toi_disposition","tfopwg_disp","disp"]
    disp_col = next((c for c in candidates if c in df.columns), None)
    if disp_col is None:
        poss = [c for c in df.columns if "disposition" in c]
        disp_col = poss[0] if poss else None
    if disp_col is None:
        print(f"⚠️ {mission}: no disposition column, columns={list(df.columns)[:20]}...")
        continue
    df["label"] = df[disp_col].map(map_label)
    df["mission"] = mission
    frames.append(df)

assert frames, "No datasets loaded. Put the CSVs in data_input/ and rerun."

merged = pd.concat(frames, ignore_index=True, sort=False)

# keep numeric features that appear in >=2 datasets (or tweak threshold)
num = merged.select_dtypes(include=[np.number])
valid_cols = [c for c in num.columns if num[c].notna().sum() > 0]
merged = merged[valid_cols + ["label","mission"]]

merged.to_csv(OUT/"merged_raw.csv", index=False)
print("✅ merged:", merged.shape, "saved to", OUT/"merged_raw.csv")
