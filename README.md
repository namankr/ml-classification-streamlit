# Fetal Health Classification — Multi-Model Comparison with Streamlit

## a. Problem Statement

Cardiotocography (CTG) is a routine, non-invasive monitoring technique used during pregnancy to record **Fetal Heart Rate (FHR)** and **Uterine Contractions (UC)**. Obstetricians interpret CTG traces to decide whether a pregnancy is progressing normally or needs urgent clinical intervention. Manual interpretation of CTG signals is time-consuming and subject to inter-observer variability.

This project builds and compares 5 classical machine learning classifiers that automatically classify a CTG exam into one of three fetal health states:

| Class | Meaning |
|---|---|
| 1 — Normal | No signs of fetal distress |
| 2 — Suspect | Ambiguous signal, warrants closer monitoring |
| 3 — Pathological | Signs of fetal distress, requires intervention |

Correctly flagging **Suspect** and **Pathological** cases (even though they are rare in the data) is the real clinical goal of such a system — missing an at-risk pregnancy is a far more costly error than a false alarm on a healthy one.

## b. Dataset Description

- **Source**: [UCI Machine Learning Repository — Cardiotocography (Dataset #193)](https://archive.ics.uci.edu/dataset/193/cardiotocography)
- **Instances**: 2,126 CTG exams
- **Features**: 21 numeric features derived from CTG signal processing (e.g. `LB` baseline fetal heart rate, `ASTV`/`ALTV` abnormal short/long term variability, `UC` uterine contraction frequency, histogram statistics like `Mean`, `Median`, `Mode`, `Variance`, etc.) — no missing values.
- **Target column**: `NSP` (Fetal State Class) — multi-class: `1 = Normal`, `2 = Suspect`, `3 = Pathological`
- **Class distribution**: 1655 Normal (78%) / 295 Suspect (14%) / 176 Pathological (8%) — a realistically **imbalanced** clinical dataset, which is why Accuracy alone is misleading and metrics like MCC and macro-averaged Precision/Recall/F1 are used (see Methodology below).
- Note: the raw UCI file also contains a `CLASS` column, a separate 10-class morphological pattern code used in other studies. It is **not** one of the 21 official input features and was dropped before training to avoid leaking label-like information.

## c. GitHub Repository Link

> `<TO BE FILLED IN AFTER YOU CREATE THE GITHUB REPO>`

## d. Models Used

### Methodology notes
- **Train/test split**: 80/20, stratified on `NSP` so both splits preserve the 78/14/8 class ratio. The test split (426 rows) is saved as [test_data.csv](./test_data.csv) and is never used during training — it is exactly what the Streamlit app evaluates.
- **Preprocessing**: every model is wrapped in a `Pipeline(StandardScaler -> Classifier)`. Scaling is mathematically required for Logistic Regression, KNN, and SVM (distance/gradient-based algorithms), and does not change predictions for Decision Tree or Naive Bayes — using one consistent pipeline shape for all 5 models keeps saved artifacts and app-loading code uniform.
- **Metric averaging**: Precision, Recall, F1, and AUC use **macro averaging** across the 3 classes (each class weighted equally regardless of how many samples it has), rather than weighted averaging. This is a deliberate choice: weighted averaging would let the dominant "Normal" class dictate the score, while macro averaging forces "Suspect" and "Pathological" performance to matter just as much — matching the real clinical priority.

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.885 | 0.961 | 0.810 | 0.760 | 0.779 | 0.683 |
| Decision Tree | 0.897 | 0.800 | 0.829 | 0.785 | 0.803 | 0.706 |
| KNN | 0.871 | 0.939 | 0.799 | 0.699 | 0.740 | 0.619 |
| Naive Bayes | 0.810 | 0.876 | 0.650 | 0.701 | 0.657 | 0.574 |
| Support Vector Machine | 0.894 | 0.965 | 0.820 | 0.753 | 0.783 | 0.697 |

*(All values computed on the held-out `test_data.csv` split, 426 samples, via `src/train_models.py`.)*

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | `<TODO>` |
| Decision Tree | `<TODO>` |
| KNN | `<TODO>` |
| Naive Bayes | `<TODO>` |
| Support Vector Machine | `<TODO>` |
| **Overall, Winner for your dataset?** | `<TODO>` |

## Project Structure

```
ml-classification-streamlit/
├── app.py                  # Streamlit app (inference + evaluation, no retraining)
├── requirements.txt
├── README.md
├── test_data.csv            # held-out test split used by the app
├── data/raw/fetal_health.csv  # full original UCI dataset
├── src/train_models.py      # trains all 5 models, saves metrics + model artifacts
└── model/                   # saved model pipelines (.joblib) + metrics.json
```

## Running Locally

```bash
pip install -r requirements.txt
python src/train_models.py   # retrains models and regenerates test_data.csv (optional - already included)
streamlit run app.py
```

## Live App

> `<TO BE FILLED IN AFTER DEPLOYING TO STREAMLIT COMMUNITY CLOUD>`
