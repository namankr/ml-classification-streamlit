"""
Train 5 classical ML classifiers on the Fetal Health Classification (CTG) dataset.

Dataset source: UCI Machine Learning Repository - Cardiotocography (ID 193)
https://archive.ics.uci.edu/dataset/193/cardiotocography

Target column: NSP (Fetal State Class)
    1 = Normal, 2 = Suspect, 3 = Pathological

Design decisions (see README.md for full rationale):
- Stratified train/test split keeps the 78/14/8 class ratio identical in both
  splits, since a random (non-stratified) split could easily under-represent
  the rare "Pathological" class in the test set.
- Every model is wrapped in a Pipeline(StandardScaler -> Classifier). Scaling
  is mathematically necessary for Logistic Regression, KNN and SVM (they are
  distance/gradient based), and does not change predictions for Decision
  Tree or Gaussian Naive Bayes. Using the same pipeline shape for all 5
  models keeps the saved artifacts and the Streamlit app's loading code
  simple and uniform.
- Precision/Recall/F1/AUC use **macro averaging** across the 3 classes.
  Macro averaging treats "Suspect" and "Pathological" as equally important
  as "Normal", which matches the real clinical goal: catching rare at-risk
  cases matters more than overall accuracy on the easy majority class.
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

RANDOM_STATE = 42
TEST_SIZE = 0.2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fetal_health.csv"
MODEL_DIR = PROJECT_ROOT / "model"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
METRICS_PATH = PROJECT_ROOT / "model" / "metrics.json"

TARGET_COL = "NSP"
# 'CLASS' is an alternate, finer-grained diagnostic code (not one of the 21
# official input features) - it must be dropped to avoid leaking label info.
DROP_COLS = ["CLASS"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(RAW_DATA_PATH)
    df = df.drop(columns=DROP_COLS)
    return df


def get_models() -> dict[str, Pipeline]:
    """Return the 5 required classifiers, each wrapped in a scaling pipeline."""
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "decision_tree": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "naive_bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GaussianNB()),
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]),
    }


def evaluate(y_true, y_pred, y_proba, classes) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro", labels=classes
        ),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
    }


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Persist the held-out test split exactly as required by the assignment:
    # the Streamlit app must evaluate saved models on this file WITHOUT
    # retraining, so it must never have been seen during training.
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"Saved held-out test data -> {TEST_DATA_PATH} ({len(test_df)} rows)")

    classes = sorted(y.unique())
    all_metrics = {}

    for name, pipeline in get_models().items():
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)

        metrics = evaluate(y_test, y_pred, y_proba, classes)
        all_metrics[name] = metrics

        model_path = MODEL_DIR / f"{name}.joblib"
        joblib.dump(pipeline, model_path)

        print(f"\n{name}")
        for metric_name, value in metrics.items():
            print(f"  {metric_name:10s}: {value:.4f}")
        print(f"  saved -> {model_path}")

    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nAll metrics saved -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
