
---

## 🐍 credit_risk_model.py (Code)  

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# Load dataset
df = pd.read_csv("german_credit_data.csv")

# Preprocessing
df = df.dropna()
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Risk_good", axis=1)  # Assuming target encoded as Risk_good
y = df["Risk_good"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)
y_proba_lr = log_reg.predict_proba(X_test)[:,1]
auc_lr = roc_auc_score(y_test, y_proba_lr)

# Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:,1]
auc_rf = roc_auc_score(y_test, y_proba_rf)

# Print results
with open("outputs/results.txt", "w") as f:
    f.write("Logistic Regression ROC-AUC: {:.2f}\n".format(auc_lr))
    f.write("Random Forest ROC-AUC: {:.2f}\n".format(auc_rf))
    f.write("\nClassification Report (Random Forest):\n")
    f.write(classification_report(y_test, y_pred_rf))

print("Logistic Regression ROC-AUC:", auc_lr)
print("Random Forest ROC-AUC:", auc_rf)

# ROC Curve
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

plt.figure(figsize=(8,6))
plt.plot(fpr_lr, tpr_lr, label="Logistic Regression (AUC = {:.2f})".format(auc_lr))
plt.plot(fpr_rf, tpr_rf, label="Random Forest (AUC = {:.2f})".format(auc_rf))
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Credit Risk Models")
plt.legend()
plt.savefig("outputs/model_performance.png")
plt.show()
