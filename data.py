import warnings
warnings.filterwarnings("once") 

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


# 1) Load data
df = pd.read_csv("datalog.csv")

print("Ukuran awal dataframe:", df.shape)
print("Kolom:", df.columns.tolist())

# 2) Basic cleaning
df.columns = df.columns.str.strip()

# Hapus kolom 'Unnamed' bila ada
if 'Unnamed: 4' in df.columns:
    df = df.drop(columns=['Unnamed: 4'])

# Hapus baris yang kosong semua
df = df.dropna(how='all')


# 3) gunakan dayfirst=True karena format di file dd/mm/YYYY
df['VOYAGE'] = pd.to_datetime(df['VOYAGE'], errors='coerce', dayfirst=True)
df['ESTIMASI ARRIVAL(ETA)'] = pd.to_datetime(df['ESTIMASI ARRIVAL(ETA)'], errors='coerce', dayfirst=True)
df['ESTIMASI DEPARTURE (ETD)'] = pd.to_datetime(df['ESTIMASI DEPARTURE (ETD)'], errors='coerce', dayfirst=True)
df['CLOSING TIME'] = pd.to_datetime(df['CLOSING TIME'], errors='coerce', dayfirst=True)

df_original = df.copy()

# 4) Buang baris yang tidak punya waktu penting (VOYAGE atau ETA)
df = df.dropna(subset=['VOYAGE', 'ESTIMASI ARRIVAL(ETA)']).reset_index(drop=True)
print("Ukuran setelah drop VOYAGE/ETA NaT:", df.shape)

# 5) Feature engineering
# DELIVERY_DELAY_MIN = ETA - VOYAGE (menit)
df['DELIVERY_DELAY_MIN'] = (df['ESTIMASI ARRIVAL(ETA)'] - df['VOYAGE']).dt.total_seconds() / 60

# Tambah DAY_OF_WEEK dan MONTH
df['DAY_OF_WEEK'] = df['VOYAGE'].dt.dayofweek
df['MONTH'] = df['VOYAGE'].dt.month

# Normalisasi nama kolom 'OPEN STACK' (gabungkan spasi dsb.)
if 'OPEN STACK' not in df.columns and 'OPEN STACK ' in df.columns:
    df.rename(columns={'OPEN STACK ': 'OPEN STACK'}, inplace=True)

# Pastikan OPEN STACK numeric
df['OPEN STACK'] = pd.to_numeric(df['OPEN STACK'], errors='coerce')

# Encode STATUS ke numerik (pastikan huruf kapital/spasi sesuai)
df['STATUS'] = df['STATUS'].astype(str).str.strip()  # normalisasi teks
df['STATUS_NUM'] = df['STATUS'].map({'On Time': 0, 'Delayed': 1})

# Jika ada status lain (typo), tampilkan untuk inspeksi
unknown_status = df.loc[df['STATUS_NUM'].isna(), 'STATUS'].unique()
if len(unknown_status) > 0:
    print("Baris dengan STATUS tidak dikenali (akan di-drop):", unknown_status)

# Drop baris yang tidak memiliki label STATUS (target harus lengkap)
df = df.dropna(subset=['STATUS_NUM']).reset_index(drop=True)
df['STATUS_NUM'] = df['STATUS_NUM'].astype(int)

print("Ukuran setelah drop STATUS NaN:", df.shape)

# 6) Inspect & optionally remove extreme outliers (optional)-
print("\nRingkasan DELIVERY_DELAY_MIN:")
print(df['DELIVERY_DELAY_MIN'].describe())

thr = df['DELIVERY_DELAY_MIN'].quantile(0.99)  # mis: potong 1% paling ekstrem
df = df[df['DELIVERY_DELAY_MIN'].between(df['DELIVERY_DELAY_MIN'].quantile(0.01), thr)].reset_index(drop=True)
print("Ukuran setelah trim outliers:", df.shape)

# 7) Statistik deskriptif & korelasi
print("\nStatistik Deskriptif (fitur utama):")
print(df[['DELIVERY_DELAY_MIN', 'OPEN STACK', 'DAY_OF_WEEK', 'MONTH', 'STATUS_NUM']].describe())

print("\nKorelasi:")
print(df[['DELIVERY_DELAY_MIN', 'OPEN STACK', 'DAY_OF_WEEK', 'MONTH', 'STATUS_NUM']].corr())

# 8) Visualisasi EDA 
plt.figure(figsize=(7,4))
sns.histplot(df['DELIVERY_DELAY_MIN'].dropna(), bins=20, kde=True)
plt.title("Distribusi Delivery Delay (Menit)")
plt.xlabel("Delay (Menit)")
plt.tight_layout()
plt.savefig("fig_hist_delay.png", dpi=200)
plt.show()

plt.figure(figsize=(9,6))
sns.scatterplot(
    data=df,
    x='DELIVERY_DELAY_MIN',
    y='OPEN STACK',
    hue='STATUS_NUM',
    palette={0: 'tab:green', 1: 'tab:red'},
    alpha=0.75,
    edgecolor='w',
    s=80
)
plt.title("Scatter: Delay vs Open Stack (warna = Status)")
plt.xlabel("Delivery Delay (menit)")
plt.ylabel("Open Stack")
plt.legend(title='Status', labels=['On Time','Delayed'])
plt.tight_layout()
plt.savefig("fig_scatter_delay_openstack.png", dpi=200)
plt.show()

# 9) Modeling preparation
features = ['DELIVERY_DELAY_MIN', 'OPEN STACK', 'DAY_OF_WEEK', 'MONTH']
X = df[features].copy()
y = df['STATUS_NUM'].copy()

# Impute missing values in X 
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Final check: tidak boleh ada NaN/inf
if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
    raise ValueError("Masih ada NaN atau inf di X_scaled — periksa imputasi/konversi numerik.")

if y.isna().any():
    raise ValueError("Masih ada NaN di y — periksa mapping STATUS -> STATUS_NUM.")

# 10) Train / Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# 11) Modeling: Logistic Regression
lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("Confusion Matrix (LR):\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report (LR):\n", classification_report(y_test, y_pred_lr))

# 12) Modeling: Random Forest
rf = RandomForestClassifier(random_state=42, n_estimators=200)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Confusion Matrix (RF):\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report (RF):\n", classification_report(y_test, y_pred_rf))

# 13) Feature importance
importances = rf.feature_importances_
plt.figure(figsize=(6,4))
sns.barplot(x=importances, y=features)
plt.title("Feature Importance (Random Forest)")
plt.tight_layout()
plt.savefig("fig_feature_importance.png", dpi=200)
plt.show()

# 14) ROC curves & AUC
y_proba_lr = lr.predict_proba(X_test)[:,1]
y_proba_rf = rf.predict_proba(X_test)[:,1]
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
plt.tight_layout()
plt.savefig("fig_roc.png", dpi=200)
plt.show()

# 15) Cross-validation 
scores_lr = cross_val_score(lr, X_scaled, y, cv=5, scoring='accuracy')
scores_rf = cross_val_score(rf, X_scaled, y, cv=5, scoring='accuracy')
print("\nLogistic Regression CV scores:", scores_lr)
print("Mean CV accuracy (Logistic Regression):", scores_lr.mean())
print("Random Forest CV scores:", scores_rf)
print("Mean CV accuracy (Random Forest):", scores_rf.mean())

# 16) Save cleaned data & models
df.to_csv("cleaned_datalog.csv", index=False)
print("\nSaved cleaned_datalog.csv and figures (fig_hist_delay.png, fig_scatter_delay_openstack.png, fig_feature_importance.png, fig_roc.png).")