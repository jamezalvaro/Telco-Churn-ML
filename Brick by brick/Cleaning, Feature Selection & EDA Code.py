import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
custom_palette = ["#3498db", "#e74c3c"] # Blue for No Churn, Red for Churn

# Data Preparation
print("Load Dataset")
file_path = 'Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv'

# Display initial dataset information
print("--- Check Dataset ---")
df = pd.read_csv(file_path)
df.info()

#PRE-SPLIT
#A. FEATURE SELECTION
#Select only 7 core features and the target (Churn)
selected_features = [
    'Contract', 'InternetService', 'TotalCharges', 'tenure', 
    'PaperlessBilling', 'MultipleLines', 'StreamingMovies', 'Churn'
]
df_clean = df[selected_features].copy()

#B. DATA CLEANING
#Convert 'TotalCharges' to numeric, replacing errors with NaN
df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

# Fill missing values (NaN) with median
median_value = df_clean['TotalCharges'].median()
df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(median_value)

print("Feature Selection & Data Cleaning completed. Ready for EDA.\n")


#EDA
print("Rendering EDA visualizations...")

#Create main figure layout
fig = plt.figure(figsize=(18, 15))
fig.suptitle('Exploratory Data Analysis (EDA) - Telco Customer Churn', fontsize=20, fontweight='bold', y=0.98)

# --- GRAPH 1: Target Distribution (Univariate) ---
ax1 = plt.subplot(2, 3, 1)
df_clean['Churn'].value_counts().plot.pie(
    autopct='%1.1f%%', colors=custom_palette, startangle=90, 
    explode=(0, 0.1), shadow=True, ax=ax1, textprops={'fontsize': 12, 'color': 'white', 'weight': 'bold'}
)
ax1.set_title('Target Class Distribution (Churn)', fontsize=14, fontweight='bold')
ax1.set_ylabel('')

# --- GRAPH 2 & 3: Numeric Features vs Target (Bivariate) ---
ax2 = plt.subplot(2, 3, 2)
sns.kdeplot(data=df_clean, x='tenure', hue='Churn', fill=True, palette=custom_palette, ax=ax2, common_norm=False)
ax2.set_title('Tenure vs Churn Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Tenure (Months)')

ax3 = plt.subplot(2, 3, 3)
sns.kdeplot(data=df_clean, x='TotalCharges', hue='Churn', fill=True, palette=custom_palette, ax=ax3, common_norm=False)
ax3.set_title('Total Charges vs Churn Distribution', fontsize=14, fontweight='bold')
ax3.set_xlabel('Total Charges')

# --- GRAPH 4, 5, 6: Categorical Features vs Target (Bivariate) ---
ax4 = plt.subplot(2, 3, 4)
sns.countplot(data=df_clean, x='Contract', hue='Churn', palette=custom_palette, ax=ax4)
ax4.set_title('Contract vs Churn', fontsize=14, fontweight='bold')

ax5 = plt.subplot(2, 3, 5)
sns.countplot(data=df_clean, x='InternetService', hue='Churn', palette=custom_palette, ax=ax5)
ax5.set_title('Internet Service vs Churn', fontsize=14, fontweight='bold')

ax6 = plt.subplot(2, 3, 6)
sns.countplot(data=df_clean, x='PaperlessBilling', hue='Churn', palette=custom_palette, ax=ax6)
ax6.set_title('Paperless Billing vs Churn', fontsize=14, fontweight='bold')

# Adjust layout
plt.tight_layout(pad=3.0)
plt.show()