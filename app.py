"""
Streamlit app for the Fetal Health Classification project.

Loads 5 pre-trained classifiers (Logistic Regression, Decision Tree, KNN,
Naive Bayes, SVM) from the model/ directory and evaluates them on a
user-supplied test CSV - it never retrains anything. This mirrors how a
model would be used in production: train once offline, then serve/evaluate
repeatedly on new data.
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
DEFAULT_TEST_DATA = PROJECT_ROOT / "test_data.csv"

TARGET_COL = "NSP"
FEATURE_COLS = [
    "LB", "AC", "FM", "UC", "DL", "DS", "DP", "ASTV", "MSTV", "ALTV",
    "MLTV", "Width", "Min", "Max", "Nmax", "Nzeros", "Mode", "Mean",
    "Median", "Variance", "Tendency",
]
CLASS_LABELS = {1: "Normal", 2: "Suspect", 3: "Pathological"}

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


@st.cache_resource
def load_models() -> dict:
    """Load all 5 saved pipelines once and cache them across reruns."""
    return {name: joblib.load(MODEL_DIR / fname) for name, fname in MODEL_FILES.items()}


def load_test_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv(DEFAULT_TEST_DATA)


def validate_columns(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems, empty if the file is valid."""
    problems = []
    missing_features = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        problems.append(f"Missing required feature columns: {missing_features}")
    if TARGET_COL not in df.columns:
        problems.append(f"Missing target column: '{TARGET_COL}'")
    return problems


def compute_metrics(model, X: pd.DataFrame, y: pd.Series) -> dict:
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    classes = sorted(y.unique())
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_proba, multi_class="ovr", average="macro", labels=classes),
        "Precision": precision_score(y, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y, y_pred, average="macro", zero_division=0),
        "F1 Score": f1_score(y, y_pred, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def plot_confusion_matrix(y_true, y_pred, classes):
    labels = [CLASS_LABELS[c] for c in classes]
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def render_single_model_view(model_name, model, X, y):
    st.subheader(f"Results: {model_name}")

    metrics = compute_metrics(model, X, y)
    cols = st.columns(6)
    for col, (metric_name, value) in zip(cols, metrics.items()):
        col.metric(metric_name, f"{value:.3f}")

    y_pred = model.predict(X)
    classes = sorted(y.unique())

    col_left, col_right = st.columns([1, 1.3])
    with col_left:
        st.markdown("**Confusion Matrix**")
        st.pyplot(plot_confusion_matrix(y, y_pred, classes))

    with col_right:
        st.markdown("**Classification Report**")
        target_names = [CLASS_LABELS[c] for c in classes]
        report = classification_report(
            y, y_pred, labels=classes, target_names=target_names,
            output_dict=True, zero_division=0,
        )
        report_df = pd.DataFrame(report).transpose().round(3)
        st.dataframe(report_df, width="stretch")


def render_compare_all_view(models, X, y):
    st.subheader("Comparison Across All 5 Models")

    rows = []
    for name, model in models.items():
        metrics = compute_metrics(model, X, y)
        metrics["Model"] = name
        rows.append(metrics)

    compare_df = pd.DataFrame(rows).set_index("Model")
    compare_df = compare_df[["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"]]

    st.dataframe(
        compare_df.style.highlight_max(axis=0, color="lightgreen").format("{:.3f}"),
        width="stretch",
    )

    st.markdown("**Metric comparison chart**")
    st.bar_chart(compare_df)


def main():
    st.set_page_config(page_title="Fetal Health Classification", layout="wide")
    st.title("Fetal Health Classification — Model Evaluation Dashboard")
    st.caption(
        "Evaluate 5 pre-trained classifiers (Logistic Regression, Decision Tree, "
        "KNN, Naive Bayes, SVM) on Cardiotocogram (CTG) test data. "
        "Models are loaded as-is; nothing is retrained here."
    )

    with st.sidebar:
        st.header("Controls")
        uploaded_file = st.file_uploader(
            "Upload test CSV (must contain the 21 CTG feature columns + 'NSP' target)",
            type=["csv"],
        )
        model_choice = st.selectbox(
            "Select model",
            ["Compare All Models"] + list(MODEL_FILES.keys()),
        )

    df = load_test_data(uploaded_file)

    if uploaded_file is None:
        st.info(
            "No file uploaded — showing results on the bundled `test_data.csv` "
            "(the held-out split never seen during training). Upload your own "
            "CSV in the sidebar to evaluate on different data."
        )

    problems = validate_columns(df)
    if problems:
        st.error("Uploaded file is not valid:\n\n" + "\n".join(f"- {p}" for p in problems))
        st.stop()

    with st.expander("Preview uploaded data", expanded=False):
        st.dataframe(df.head(20), width="stretch")
        st.caption(f"{len(df)} rows total")

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    models = load_models()

    if model_choice == "Compare All Models":
        render_compare_all_view(models, X, y)
    else:
        render_single_model_view(model_choice, models[model_choice], X, y)


if __name__ == "__main__":
    main()
