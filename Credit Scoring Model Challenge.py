import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import joblib

# Load data
train_df = pd.read_csv("cs-training.csv", index_col=0)
test_df = pd.read_csv("cs-test.csv", index_col=0)

# Separate features and target
X = train_df.drop("SeriousDlqin2yrs", axis=1)
y = train_df["SeriousDlqin2yrs"]

# Basic preprocessing
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
X_test_imputed = imputer.transform(test_df)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(X_imputed, y, test_size=0.2, stratify=y, random_state=42)

# Model training
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='auc',
    random_state=42
)
model.fit(X_train, y_train)

# Evaluation
y_pred_proba = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred_proba)
print(f"AUC on validation set: {auc:.4f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Predict on test set
test_preds = model.predict_proba(X_test_imputed)[:, 1]
test_output = pd.DataFrame({
    "Id": test_df.index,
    "Probability": test_preds
})
test_output.to_csv("credit_scoring_predictions.csv", index=False)

# Save model
joblib.dump(model, "credit_scoring_model.pkl")
joblib.dump(imputer, "imputer.pkl")

print("Model and predictions saved.")
