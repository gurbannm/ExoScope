# src/exoscope_app.py
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# =========================
# --------- SETUP ---------
# =========================

st.set_page_config(
    page_title="ExoScope: AI-Powered Exoplanet Classifier",
    page_icon="🔭",
    layout="wide",
)

st.title("🔭 ExoScope: AI-Powered Exoplanet Classifier")
st.caption("Explore and classify exoplanets from NASA’s TESS, K2, and Cumulative datasets with machine learning.")


# --------- helpers ---------
@st.cache_resource
def load_bundle(path: Path):
    """Load a saved model bundle: {'model','scaler','features','classes'}."""
    return joblib.load(path)

def read_csv_lazy(p: Path) -> pd.DataFrame:
    if not p.exists():
        st.warning(f"Missing file: {p}")
        return pd.DataFrame()
    return pd.read_csv(p)

def load_metrics(p: Path):
    if not p.exists():
        return None
    with open(p, "r") as f:
        return json.load(f)

def confusion_dataframe(cm: list, classes: list[str]) -> pd.DataFrame:
    """Return long-format DF for heatmap from confusion-matrix list."""
    arr = np.array(cm)
    df = pd.DataFrame(arr, index=classes, columns=classes)
    df = df.reset_index().melt(id_vars="index", var_name="Predicted", value_name="Count")
    df.rename(columns={"index": "True"}, inplace=True)
    return df

def plot_confusion_altair(cm_df: pd.DataFrame):
    chart = (
        alt.Chart(cm_df)
        .mark_rect()
        .encode(
            x=alt.X("Predicted:N", sort=cm_df["Predicted"].unique().tolist()),
            y=alt.Y("True:N", sort=cm_df["True"].unique().tolist()),
            color=alt.Color("Count:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["True:N", "Predicted:N", "Count:Q"],
        )
        .properties(height=300)
    )
    return chart

def local_top_drivers(clf, scaler, features, x_row: pd.DataFrame, k: int = 8, delta: float = 0.1):
    """
    Local sensitivity: nudge each feature up/down a bit and see how
    the top-class probability changes. (Fast, model-agnostic.)
    """
    x_s = scaler.transform(x_row)
    base = clf.predict_proba(x_s)[0]
    target = int(np.argmax(base))
    impacts = []
    for f in features:
        xm = x_row.copy()
        v = xm.iloc[0][f]
        step = abs(v) * delta if abs(v) > 1e-9 else 1.0
        for sgn in (+1, -1):
            xm.iloc[0, xm.columns.get_loc(f)] = v + sgn * step
            p = clf.predict_proba(scaler.transform(xm))[0][target]
            impacts.append((f, p - base[target]))
    df_imp = pd.DataFrame(impacts, columns=["feature", "delta"])
    local = (
        df_imp.groupby("feature")["delta"]
        .apply(lambda s: s.abs().max())
        .sort_values(ascending=False)
        .head(k)
        .rename("impact")
        .reset_index()
    )
    return local


# =========================
# ----- SIDEBAR / IO ------
# =========================

st.sidebar.header("Model")

model_choice = st.sidebar.selectbox(
    "Choose a trained model",
    ["TESS only", "Merged", "Merged (tuned)"],
    index=1  # default to Merged; change to 2 to prefer tuned
)

# Map choice -> files
if model_choice == "TESS only":
    model_path = Path("models/planet_classifier.joblib")
    proc_dir = Path("data_input/processed")
    metrics_js = Path("reports/metrics.json")
    feats_csv = Path("reports/top_features_tess.csv")
elif model_choice == "Merged":
    model_path = Path("models/planet_classifier_merged.joblib")
    proc_dir = Path("data_input/processed_merged")
    metrics_js = Path("reports/metrics_merged.json")
    feats_csv = Path("reports/top_features_merged.csv")
else:  # Merged (tuned)
    model_path = Path("models/planet_classifier_merged_tuned.joblib")
    proc_dir = Path("data_input/processed_merged")
    metrics_js = Path("reports/metrics_merged_tuned.json")
    feats_csv = Path("reports/top_features_merged_tuned.csv")

# load model
bundle = load_bundle(model_path)
clf = bundle["model"]
scaler = bundle["scaler"]
features: list[str] = bundle["features"]
classes: list[str] = bundle["classes"]

# load processed test set (for browsing & truth check)
X_test = read_csv_lazy(proc_dir / "X_test.csv")
y_test = read_csv_lazy(proc_dir / "y_test.csv").squeeze("columns") if (proc_dir / "y_test.csv").exists() else None

# row browser
max_idx = max(0, len(X_test) - 1)
idx = st.sidebar.number_input("Row index from X_test", min_value=0, max_value=max_idx if max_idx > 0 else 0, value=0, step=1)

# upload CSV
uploaded = st.sidebar.file_uploader("Or upload CSV (same feature columns)", type=["csv"])

st.sidebar.caption("Drag-and-drop or browse a CSV with the same numeric feature columns used by the model.")


# =========================
# --------- TABS ----------
# =========================

tab_pred, tab_metrics, tab_about = st.tabs(["🔎 Prediction", "📈 Test metrics", "ℹ️ About"])

# ---------- PREDICTION TAB ----------
with tab_pred:
    colL, colM, colR = st.columns([1.1, 0.9, 1.2])

    # figure out the input row / uploaded row
    source = None
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            # check schema
            missing = [c for c in features if c not in df_up.columns]
            if missing:
                st.error(f"Uploaded CSV missing columns: {missing[:8]}{' ...' if len(missing) > 8 else ''}")
                st.stop()
            x_df = df_up[features].iloc[[0]]  # first row
            source = "Uploaded CSV (row 0)"
        except Exception as e:
            st.error(f"Could not read uploaded file: {e}")
            st.stop()
    else:
        if len(X_test) == 0:
            st.warning("Test set not found. Please ensure processed data exists.")
            st.stop()
        x_df = X_test.iloc[[idx]][features]
        source = f"X_test row #{idx}"

    # scale & predict
    x_s = scaler.transform(x_df)
    proba = clf.predict_proba(x_s)[0]
    pred_class = classes[int(np.argmax(proba))]

    with colL:
        st.subheader("Prediction")
        st.write(f"**Source:** {source}")
        st.metric("Predicted class", pred_class)

        # Ground truth (if available)
        if y_test is not None and uploaded is None and idx < len(y_test):
            true_label = str(y_test.iloc[idx])
            if true_label == pred_class:
                st.success(f"✅ Ground truth: **{true_label}** (match)")
            else:
                st.error(f"❌ Ground truth: **{true_label}** (differs)")

        # Batch scoring if uploaded has multiple rows
        if uploaded is not None and len(df_up) > 1:
            st.divider()
            st.subheader("Batch predictions (uploaded)")
            Xb = df_up[features].copy()
            Xb_s = scaler.transform(Xb)
            preds = clf.predict(Xb_s)
            probs = clf.predict_proba(Xb_s)
            out = df_up.copy()
            out["pred_class"] = preds
            for i, c in enumerate(classes):
                out[f"proba_{c}"] = probs[:, i]
            st.dataframe(out.head(25))
            csv_bytes = out.to_csv(index=False).encode()
            st.download_button("Download predictions CSV", data=csv_bytes, file_name="exoscope_predictions.csv")

    with colM:
        st.subheader("Probabilities")
        prob_df = pd.DataFrame({"class": classes, "probability": proba})
        chart = (
            alt.Chart(prob_df)
            .mark_bar()
            .encode(
                x=alt.X("class:N", sort=classes),
                y=alt.Y("probability:Q"),
                tooltip=["class", alt.Tooltip("probability:Q", format=".3f")],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    with colR:
        # GLOBAL vs LOCAL importance
        mode = st.radio("Importance mode", ["Global (overall)", "Local (this row)"], horizontal=True)
        if mode == "Global (overall)":
            st.subheader("Top features")
            feat_df = read_csv_lazy(feats_csv)
            if not feat_df.empty and {"feature", "importance"}.issubset(feat_df.columns):
                chart2 = (
                    alt.Chart(feat_df.head(20))
                    .mark_bar()
                    .encode(
                        x=alt.X("importance:Q"),
                        y=alt.Y("feature:N", sort="-x"),
                        tooltip=["feature", alt.Tooltip("importance:Q", format=".4f")],
                    )
                    .properties(height=340)
                )
                st.altair_chart(chart2, use_container_width=True)
                st.caption("Permutation importance on the test set (higher = more influential overall).")
            else:
                st.info("Top-features file not found or invalid.")
        else:
            st.subheader("Top drivers for **this** prediction")
            local = local_top_drivers(clf, scaler, features, x_df, k=8)
            if not local.empty:
                chart3 = (
                    alt.Chart(local)
                    .mark_bar()
                    .encode(
                        x=alt.X("impact:Q"),
                        y=alt.Y("feature:N", sort="-x"),
                        tooltip=["feature", alt.Tooltip("impact:Q", format=".4f")],
                    )
                    .properties(height=340)
                )
                st.altair_chart(chart3, use_container_width=True)
                st.caption("Change in predicted-class probability if each feature were nudged up/down slightly.")

# ---------- METRICS TAB ----------
with tab_metrics:
    st.subheader("Test metrics")

    metrics = load_metrics(metrics_js)
    if metrics is None:
        st.info("Metrics file not found. Train the model and generate reports to see metrics.")
    else:
        rep = metrics.get("report", {})
        # accuracy
        acc = rep.get("accuracy", None)
        if acc is not None:
            st.metric("Accuracy", f"{acc:.3f}")
        # per-class table
        rows = []
        for label in ["False Positive", "Planet", "Candidate"]:
            if label in rep:
                r = rep[label]
                rows.append({
                    "class": label,
                    "precision": float(r.get("precision", 0)),
                    "recall": float(r.get("recall", 0)),
                    "f1": float(r.get("f1-score", 0)),
                })
        if rows:
            df_rows = pd.DataFrame(rows)
            st.dataframe(df_rows, use_container_width=True)

        # confusion matrix
        cm = metrics.get("confusion_matrix", None)
        if cm is not None:
            st.subheader("Confusion matrix (test)")
            cm_df = confusion_dataframe(cm, classes)
            st.altair_chart(plot_confusion_altair(cm_df), use_container_width=True)

    st.caption("Model: HistGradientBoosting | Data: TESS, K2, Cumulative | Created for NASA Space Apps Challenge 2025")

# ---------- ABOUT TAB ----------
with tab_about:
    st.markdown(
        """
### What is ExoScope?
ExoScope is a machine-learning classifier that predicts whether a TESS/K2 candidate is a **Planet**, **Candidate**, or **False Positive**.
It’s trained on NASA open datasets (TESS, K2, and the cumulative exoplanet archive).

**How to use**
- Pick a **model** in the sidebar (TESS / Merged / Merged tuned).
- Move the **row index** to browse the held-out test set, or upload your own CSV with the same feature columns.
- See the **prediction**, **class probabilities**, and **feature importance**.
- In **Test metrics**, view overall accuracy, per-class metrics, and the confusion matrix.

**Notes**
- *Global* feature importance: which variables the model relied on overall (permutation importance).
- *Local* drivers: which variables most affected the current row’s prediction (sensitivity analysis).
        """
    )
