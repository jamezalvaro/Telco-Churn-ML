import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split, StratifiedKFold, learning_curve, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import clone
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# --- System Configuration ---
sns.set_theme(style="whitegrid")

print("System Initializing...")
print("Loading Dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv")
file_path = 'Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv' 
df = pd.read_csv(file_path)


# --- Data Preparation (Pre-Split) ---
print("Data Preparation...")
selected_features = [
    'Contract', 'InternetService', 'TotalCharges', 'tenure', 
    'PaperlessBilling', 'MultipleLines', 'StreamingMovies', 'Churn'
]
df_clean = df[selected_features].copy()

df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(df_clean['TotalCharges'].median())


# --- Target Encoding & Data Split ---
print("Target Encoding & Data Split (80:20)...")
X = df_clean.drop('Churn', axis=1)
y = df_clean['Churn'].map({'No': 0, 'Yes': 1}) 

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# --- Initialization of Preprocessor & SMOTE Rules ---
numeric_features = ['tenure', 'TotalCharges']
categorical_features = ['Contract', 'InternetService', 'PaperlessBilling', 'MultipleLines', 'StreamingMovies']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

smote = SMOTE(random_state=42)


# --- Hyperparameter Tuning & Validation ---
print("Hyperparameter Tuning Process...")

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
        scoring='accuracy', n_jobs=-1, random_state=42
    )
    search.fit(X_train, y_train)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    tuning_times[name] = elapsed_time
    
    best_pipelines[name] = search.best_estimator_
    print(f"    Completed in {elapsed_time:.2f} seconds.")
    print(f"    Optimal Parameters: {search.best_params_}")


# --- Gap Analysis & Learning Curves Rendering ---
print("\nGenerating Gap Analysis & Learning Curves...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Validation Diagnostics: Tuned Learning Curves', fontsize=16, fontweight='bold')

for i, (name, pipe) in enumerate(best_pipelines.items()):
    train_sizes, train_scores, val_scores = learning_curve(
        pipe, X_train, y_train, cv=cv_strategy, 
        train_sizes=np.linspace(0.1, 1.0, 5), scoring='accuracy', n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    gap = train_mean[-1] - val_mean[-1]
    
    status = "EXCELLENT" if gap < 0.03 else "GOOD FIT" if gap < 0.06 else "OVERFITTING"
    print(f"\nDiagnostic Report: {name.upper()}")
    print(f" -> Final Training Accuracy   : {train_mean[-1]:.4f}")
    print(f" -> Final Validation Accuracy : {val_mean[-1]:.4f}")
    print(f" -> Performance Gap           : {gap:.4f} ({status})")
    
    axes[i].plot(train_sizes, train_mean, 'o-', color="r", label="Training Score")
    axes[i].plot(train_sizes, val_mean, 'o-', color="g", label="Validation Score")
    axes[i].set_title(f"{name} ({status} | Gap: {gap:.4f})")
    axes[i].set_xlabel("Training Set Size")
    axes[i].set_ylabel("Accuracy Metric")
    axes[i].legend(loc='lower right')

plt.tight_layout()
plt.show()


# EXPLICIT FINAL TRAINING AND EVALUATION

print(" EXPLICIT FINAL TRAINING AND EVALUATION REPORT ")

# --- Training Data Preprocessing ---
print("Fitting and transforming 80% training data...")
X_train_prep = preprocessor.fit_transform(X_train) 

# --- SMOTE (Training Data Only) ---
print("[Applying SMOTE to preprocessed training data...")
X_train_smote, y_train_smote = smote.fit_resample(X_train_prep, y_train)

# --- Testing Data Preprocessing ---
print("Transforming 20% unseen testing data...")
X_test_prep = preprocessor.transform(X_test)

final_training_times = {}

for name, pipeline in best_pipelines.items():
    # Extract the tuned classifier and clone it to ensure a fresh, untrained model
    tuned_clf = pipeline.named_steps['clf']
    final_model = clone(tuned_clf) 
    
    # --- Final Model Training ---
    print(f"Training Final {name} Model on 100% SMOTE data...")
    start_time_train = time.time()
    
    final_model.fit(X_train_smote, y_train_smote)
    
    end_time_train = time.time()
    final_training_times[name] = end_time_train - start_time_train
    
    # --- Final Testing ---
    print(f"Testing Final {name} Model on Unseen Data...")
    start_time_pred = time.time()
    
    y_pred = final_model.predict(X_test_prep)
    
    end_time_pred = time.time()
    prediction_time = end_time_pred - start_time_pred
    
    acc_test = accuracy_score(y_test, y_pred)
    
    # --- Evaluation Report ---
    print(f"\n PERFORMANCE METRICS: {name.upper()} FINAL MODEL ")
    print(f"Tuning Duration         : {tuning_times[name]:.4f} seconds")
    print(f"Final Training Duration : {final_training_times[name]:.4f} seconds")
    print(f"Prediction Duration     : {prediction_time:.4f} seconds")
    print(f"Test Set Accuracy       : {acc_test * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn (0)', 'Churn (1)']))