# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.impute import SimpleImputer

# Load data
df = pd.read_csv("datalog.csv")

# --- Data Cleaning ---
df.columns = df.columns.str.strip()
critical_cols = ['VOYAGE', 'ESTIMASI ARRIVAL(ETA)']
df = df.dropna(subset=critical_cols)
df['VOYAGE'] = pd.to_datetime(df['VOYAGE'], errors='coerce')
df['ESTIMASI ARRIVAL(ETA)'] = pd.to_datetime(df['ESTIMASI ARRIVAL(ETA)'], errors='coerce')

# --- Feature Engineering ---
df['DELIVERY_DELAY_MIN'] = (df['ESTIMASI ARRIVAL(ETA)'] - df['VOYAGE']).dt.total_seconds() / 60
df['DAY_OF_WEEK'] = df['VOYAGE'].dt.dayofweek
df['MONTH'] = df['VOYAGE'].dt.month
df['STATUS_NUM'] = df['STATUS'].map({'On Time': 0, 'Delayed': 1})

# --- Statistik Deskriptif & Korelasi ---
print("\nStatistik Deskriptif:")
print(df[['DELIVERY_DELAY_MIN', 'OPEN STACK', 'DAY_OF_WEEK', 'MONTH', 'STATUS_NUM']].describe())
print("\nKorelasi antar fitur:")
print(df[['DELIVERY_DELAY_MIN', 'OPEN STACK', 'DAY_OF_WEEK', 'MONTH', 'STATUS_NUM']].corr())

# --- Visualisasi Distribusi Delay ---
plt.figure(figsize=(7,4))
sns.histplot(df['DELIVERY_DELAY_MIN'], bins=20, kde=True)
plt.title("Distribusi Delivery Delay (Menit)")
plt.xlabel("Delay (Menit)")
plt.show()

# --- Select Features for Modeling ---
features = ['DELIVERY_DELAY_MIN', 'OPEN STACK', 'DAY_OF_WEEK', 'MONTH']
X = df[features]
y = df['STATUS_NUM']

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- Modeling Logistic Regression ---
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# --- Modeling Random Forest ---
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("Random Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))

# --- Feature Importance (Random Forest) ---
importances = rf_model.feature_importances_
plt.figure(figsize=(6,4))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance (Random Forest)")
plt.show()

# --- ROC Curve untuk kedua model ---
y_proba_lr = model.predict_proba(X_test)[:,1]
y_proba_rf = rf_model.predict_proba(X_test)[:,1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_proba_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
plt.figure(figsize=(6,5))
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={roc_auc_score(y_test, y_proba_lr):.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={roc_auc_score(y_test, y_proba_rf):.2f})")
plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# --- Visualize confusion matrix (Logistic Regression) ---
plt.figure(figsize=(5,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=["On Time", "Delayed"], yticklabels=["On Time", "Delayed"])
plt.title("Confusion Matrix Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# --- Cross-validation ---
scores_lr = cross_val_score(model, X, y, cv=5)
scores_rf = cross_val_score(rf_model, X, y, cv=5)
print("Logistic Regression CV scores:", scores_lr)
print("Mean CV accuracy (Logistic Regression):", scores_lr.mean())
print("Random Forest CV scores:", scores_rf)
print("Mean CV accuracy (Random Forest):", scores_rf.mean())

# --- Save cleaned data ---
df.to_csv("cleaned_datalog.csv", index=False)