import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier


# SYSTEM INITIALIZATION & CONFIGURATION
sns.set_theme(style="whitegrid")
custom_palette = ["#3498db", "#e74c3c"] # Blue for No Churn, Red for Churn

print("System Initializing...")
print("Loading Dataset: Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv")
file_path = 'Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv' 
df = pd.read_csv(file_path)


# FEATURE SELECTION & INGESTION
print("\nExecuting Data Preparation...")
selected_features = [
    'Contract', 'InternetService', 'TotalCharges', 'tenure', 
    'PaperlessBilling', 'MultipleLines', 'StreamingMovies', 'Churn'
]
df_clean = df[selected_features].copy()

# Enforce numeric parsing on TotalCharges
df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')

# EXPLORATORY DATA ANALYSIS (EDA)
print("Rendering EDA visualizations...")
# Create a temporary dataframe for safe visualization plotting without missing values
df_eda = df_clean.copy()
df_eda['TotalCharges'] = df_eda['TotalCharges'].fillna(df_eda['TotalCharges'].median())

fig = plt.figure(figsize=(18, 15))
fig.suptitle('Exploratory Data Analysis (EDA) - Telco Customer Churn', fontsize=20, fontweight='bold', y=0.98)

# GRAPH 1: Target Distribution (Univariate)
ax1 = plt.subplot(2, 3, 1)
df_eda['Churn'].value_counts().plot.pie(
    autopct='%1.1f%%', colors=custom_palette, startangle=90, 
    explode=(0, 0.1), shadow=True, ax=ax1, textprops={'fontsize': 12, 'color': 'white', 'weight': 'bold'}
)
ax1.set_title('Target Class Distribution (Churn)', fontsize=14, fontweight='bold')
ax1.set_ylabel('')

# GRAPH 2 & 3: Numeric Features vs Target (Bivariate)
ax2 = plt.subplot(2, 3, 2)
sns.kdeplot(data=df_eda, x='tenure', hue='Churn', fill=True, palette=custom_palette, ax=ax2, common_norm=False)
ax2.set_title('Tenure vs Churn Distribution', fontsize=14, fontweight='bold')
ax2.set_xlabel('Tenure (Months)')

ax3 = plt.subplot(2, 3, 3)
sns.kdeplot(data=df_eda, x='TotalCharges', hue='Churn', fill=True, palette=custom_palette, ax=ax3, common_norm=False)
ax3.set_title('Total Charges vs Churn Distribution', fontsize=14, fontweight='bold')
ax3.set_xlabel('Total Charges')

# GRAPH 4, 5, 6: Categorical Features vs Target (Bivariate)
ax4 = plt.subplot(2, 3, 4)
sns.countplot(data=df_eda, x='Contract', hue='Churn', palette=custom_palette, ax=ax4)
ax4.set_title('Contract vs Churn', fontsize=14, fontweight='bold')

ax5 = plt.subplot(2, 3, 5)
sns.countplot(data=df_eda, x='InternetService', hue='Churn', palette=custom_palette, ax=ax5)
ax5.set_title('Internet Service vs Churn', fontsize=14, fontweight='bold')

ax6 = plt.subplot(2, 3, 6)
sns.countplot(data=df_eda, x='PaperlessBilling', hue='Churn', palette=custom_palette, ax=ax6)
ax6.set_title('Paperless Billing vs Churn', fontsize=14, fontweight='bold')

plt.tight_layout(pad=3.0)
plt.show()


# TARGET ENCODING & DATA SPLIT
print("\nExecuting Target Encoding & Data Split (80:20)...")
X = df_clean.drop('Churn', axis=1)
y = df_clean['Churn'].map({'No': 0, 'Yes': 1}) 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Calculate median strictly from training subset to isolate validation/testing data completely
train_median = X_train['TotalCharges'].median()
X_train['TotalCharges'] = X_train['TotalCharges'].fillna(train_median)
X_test['TotalCharges'] = X_test['TotalCharges'].fillna(train_median)

# PREPROCESSOR & RULE DEFINITIONS
numeric_features = ['tenure', 'TotalCharges']
categorical_features = ['Contract', 'InternetService', 'PaperlessBilling', 'MultipleLines', 'StreamingMovies']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

smote = SMOTE(random_state=42)

# HYPERPARAMETER TUNING (CV)
print("\nInitiating Hyperparameter Tuning Process...")

pipelines = {
    'Random Forest': (
        ImbPipeline([
            ('prep', preprocessor), 
            ('smote', smote), 
            ('clf', RandomForestClassifier(random_state=42, class_weight='balanced'))
        ]),
        {
            'clf__n_estimators': [100, 200, 300],
            'clf__max_depth': [4, 5, 6],            
            'clf__min_samples_leaf': [20, 50],   
            'clf__min_samples_split': [50, 100]  
        }
    ),
    'XGBoost': (
        ImbPipeline([
            ('prep', preprocessor), 
            ('smote', smote), 
            ('clf', XGBClassifier(random_state=42, eval_metric='logloss'))
        ]),
        {
            'clf__max_depth': [3, 4, 5],            
            'clf__learning_rate': [0.05, 0.1],        
            'clf__reg_alpha': [0.1, 1, 5],        
            'clf__reg_lambda': [0.1, 1, 5],       
            'clf__gamma': [0.5, 1, 2]            
        }
    )
}

best_pipelines = {}
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
tuning_times = {}

for name, (pipe, params) in pipelines.items():
    print(f" -> Tuning process started for {name}...")
    start_time = time.time()
    
    search = RandomizedSearchCV(
        pipe, params, n_iter=5, cv=cv_strategy, 
        scoring='f1', n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    tuning_times[name] = elapsed_time
    
    best_pipelines[name] = search.best_estimator_
    print(f"    Completed in {elapsed_time:.2f} seconds.")
    print(f"    Optimal Parameters Found: {search.best_params_}")


# GAP ANALYSIS & DIAGNOSTICS
print("\nGenerating Gap Analysis & Learning Curves...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Validation Diagnostics: Tuned Learning Curves (F1-Score)', fontsize=16, fontweight='bold')

for i, (name, pipe) in enumerate(best_pipelines.items()):
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X_train, y_train, cv=cv_strategy, 
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='f1', n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    gap = train_mean[-1] - val_mean[-1]
    
    status = "EXCELLENT" if gap < 0.03 else "GOOD FIT" if gap < 0.06 else "OVERFITTING"
    print(f"\nDiagnostic Report: {name.upper()}")
    print(f" -> Final Training F1-Score   : {train_mean[-1]:.4f}")
    print(f" -> Final Validation F1-Score : {val_mean[-1]:.4f}")
    print(f" -> Performance Gap           : {gap:.4f} ({status})")
    
    axes[i].plot(train_sizes, train_mean, 'o-', color="r", label="Training Score")
    axes[i].plot(train_sizes, val_mean, 'o-', color="g", label="Validation Score")
    axes[i].set_title(f"{name} ({status} | Gap: {gap:.4f})")
    axes[i].set_xlabel("Training Set Size")
    axes[i].set_ylabel("F1 Score")
    axes[i].legend(loc='lower right')

plt.tight_layout()
plt.show()


# EXPLICIT FINAL TRAINING & TEST
print("EXPLICIT FINAL TRAINING AND METRICS REPORT")

X_train_prep = preprocessor.fit_transform(X_train) 
X_test_prep = preprocessor.transform(X_test)
X_train_smote, y_train_smote = smote.fit_resample(X_train_prep, y_train)

final_training_times = {}

for name, pipeline in best_pipelines.items():
    tuned_clf = pipeline.named_steps['clf']
    final_model = clone(tuned_clf) 
    
    start_time_train = time.time()
    final_model.fit(X_train_smote, y_train_smote)
    end_time_train = time.time()
    final_training_times[name] = end_time_train - start_time_train
    
    start_time_pred = time.time()
    y_pred = final_model.predict(X_test_prep)
    end_time_pred = time.time()
    prediction_time = end_time_pred - start_time_pred
    
    f1_test = f1_score(y_test, y_pred)
    
    print(f"\n[ PERFORMANCE METRICS: {name.upper()} FINAL MODEL ]")
    print(f"Tuning Duration         : {tuning_times[name]:.4f} seconds")
    print(f"Final Training Duration : {final_training_times[name]:.4f} seconds")
    print(f"Prediction Duration     : {prediction_time:.4f} seconds")
    print(f"Test Set F1-Score       : {f1_test:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn (0)', 'Churn (1)']))