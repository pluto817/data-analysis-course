
# Spambase Classification using SVM

from pathlib import Path

import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score
)


# 1. Load dataset
BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "spambase.data"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

data = pd.read_csv(DATA_PATH, header=None)

print("========== Dataset Overview ==========")
print("Dataset shape:", data.shape)
print("\nFirst five rows:")
print(data.head())
print("\nMissing values:", data.isnull().sum().sum())


# 2. Prepare features and target
# The first 57 columns are email features.
# The last column is the class label:
# 1 = spam, 0 = non-spam.

X = data.iloc[:, :-1]
y = data.iloc[:, -1].astype(int)

print("\nClass distribution:")
print(y.value_counts().sort_index().rename({
    0: "Non-spam",
    1: "Spam"
}))


# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 4. Build preprocessing and SVM pipeline
svm_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", SVC())
])


# 5. Tune hyperparameters using cross-validation
param_grid = {
    "classifier__C": [0.1, 1, 10],
    "classifier__gamma": ["scale", 0.01, 0.1],
    "classifier__kernel": ["rbf", "linear"]
}

grid_search = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

print("\n========== Best SVM Parameters ==========")
print(grid_search.best_params_)
print("Best cross-validation Macro F1:",
      round(grid_search.best_score_, 4))


# 6. Evaluate model
y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n========== Test Results ==========")
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    labels=[0, 1],
    target_names=["Non-spam", "Spam"],
    zero_division=0
))

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("Confusion Matrix:")
print(cm)

decision_scores = best_model.decision_function(X_test)
auc = roc_auc_score(y_test, decision_scores)

print(f"ROC-AUC: {auc:.4f}")


# 7. Save confusion matrix
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-spam", "Spam"]
)

display.plot(values_format="d")
plt.title("Spambase - SVM Confusion Matrix")
plt.tight_layout()
plt.savefig(
    RESULTS_DIR / "confusion_matrix.png",
    dpi=300
)
plt.close()


# 8. Save metrics
metrics = pd.DataFrame([{
    "Best Parameters": str(grid_search.best_params_),
    "CV Macro F1": grid_search.best_score_,
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
