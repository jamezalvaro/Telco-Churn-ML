import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder # StandardScaler dihapus

#Import yang digunakan

#Sumber file
file_path = 'Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv'

#Menampilkan Informasi awal isi Data Set
print("--- Pemeriksaan Info Data Awal ---")
df = pd.read_csv(file_path)
df.info()

# --- DATA CLEANING ---
# Menghapus identifier (customerID)
df_clean = df.drop(columns=['customerID']) 

# Memperbaiki tipe data TotalCharges menjadi numerik
df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

# Imputasi Missing Values dengan Median
imputer = SimpleImputer(strategy='median')
df_clean[['TotalCharges']] = imputer.fit_transform(df_clean[['TotalCharges']])

# --- EXPLORATORY DATA ANALYSIS (EDA) ---
# Distribusi Target Churn
print("\n--- Distribusi Churn (%) ---")
print(df_clean['Churn'].value_counts(normalize=True) * 100)

# Visualisasi: Monthly Charges vs Churn
plt.figure(figsize=(8,5))
sns.kdeplot(data=df_clean, x='MonthlyCharges', hue='Churn', fill=True, common_norm=False, palette='crest')
plt.title('Density Plot: Monthly Charges vs Churn')
plt.show()

# Visualisasi: Churn berdasarkan Layanan Internet
plt.figure(figsize=(8,5))
sns.countplot(data=df_clean, x='InternetService', hue='Churn', palette='Set2')
plt.title('Distribusi Churn berdasarkan Layanan Internet')
plt.show()

# --- TRANSFORMASI FITUR (ENCODING & SCALING) ---
# Ordinal Encoding pada kolom Contract
contract_order = [['Month-to-month', 'One year', 'Two year']]
ordinal_enc = OrdinalEncoder(categories=contract_order)
df_clean['Contract_Encoded'] = ordinal_enc.fit_transform(df_clean[['Contract']])

# Memisahkan Fitur dan Target
X = df_clean.drop(columns=['Churn', 'Contract'])
y = df_clean['Churn'].map({'No': 0, 'Yes': 1}) # Mengubah target ke angka

# One-Hot Encoding untuk sisa kategori
cat_features = X.select_dtypes(include=['object']).columns
X_enc = pd.get_dummies(X, columns=cat_features, drop_first=True)

# --- Menggunakan Min-Max Scaling ---
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_enc)
X_final = pd.DataFrame(X_scaled, columns=X_enc.columns)

# EXPORT HASIL KE FOLDER
X_final.to_csv('X_processed_telco.csv', index=False)
y.to_csv('y_processed_telco.csv', index=False)

print("\nData preprocessing selesai dan tersimpan di Folder anda!") 