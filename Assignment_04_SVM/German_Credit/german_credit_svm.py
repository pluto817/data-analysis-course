# German Credit Classification using SVM

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# 1. Load dataset
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "german.data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

column_names = [
    "checking_account", "duration", "credit_history", "purpose",
    "credit_amount", "savings", "employment", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since",
    "property", "age", "other_installment_plans", "housing",
    "existing_credits", "job", "liable_people", "telephone",
    "foreign_worker", "credit_risk"
]

data = pd.read_csv(
    DATA_PATH,
    sep=r"\s+",
    header=None,
    names=column_names
)

print("========== Dataset Overview ==========")
print("Dataset shape:", data.shape)
print("\nFirst five rows:")
print(data.head())
print("\nMissing values:")
print(data.isnull().sum())


# 2. Prepare features and target
X = data.drop(columns=["credit_risk"])
y = data["credit_risk"].map({1: 1, 2: 0})

if y.isnull().any():
    raise ValueError("Unexpected target labels. Please check german.data.")

# In this experiment:
# 1 = Good credit risk
# 0 = Bad credit risk

numeric_features = [
    "duration", "credit_amount", "installment_rate",
    "residence_since", "age", "existing_credits", "liable_people"
]

categorical_features = [
    col for col in X.columns if col not in numeric_features
]

print("\nClass distribution:")
print(y.value_counts().rename(index={1: "Good credit", 0: "Bad credit"}))


# 3. Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 4. Preprocess numerical and categorical features
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])


# 5. Build SVM pipeline
svm_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", SVC())
])


# 6. Tune SVM hyperparameters using cross-validation
param_grid = {
    "classifier__C": [0.1, 1, 10],
    "classifier__gamma": ["scale", 0.01, 0.1],
    "classifier__kernel": ["rbf", "linear"]
}

grid_search = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("\n========== Best SVM Parameters ==========")
print(grid_search.best_params_)
print("Best cross-validation F1:", round(grid_search.best_score_, 4))


# 7. Evaluate on the held-out test set
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n========== Test Results ==========")
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Bad credit", "Good credit"],
    zero_division=0
))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
print(cm)

# ROC-AUC is calculated using the model's decision scores.
decision_scores = best_model.decision_function(X_test)
auc = roc_auc_score(y_test, decision_scores)
print(f"ROC-AUC: {auc:.4f}")


# 8. Save confusion matrix plot
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Bad credit", "Good credit"]
)
display.plot(values_format="d")
plt.title("German Credit - SVM Confusion Matrix")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=300)
plt.close()


# 9. Save experiment metrics
metrics = pd.DataFrame([{
    "Best Parameters": str(grid_search.best_params_),
    "CV F1": grid_search.best_score_,
    "Test Accuracy": accuracy,
    "Test ROC-AUC": auc
}])

metrics.to_csv(
    RESULTS_DIR / "svm_metrics.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nExperiment completed.")
print("Results saved to:", RESULTS_DIR)