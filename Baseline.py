import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 1. Hardcode reference paper results
paper_metrics = {
    'Random Forest': {
        'Accuracy': 0.81,
        '0': {'Precision': 0.79, 'Recall': 0.82, 'F1': 0.81},
        '1': {'Precision': 0.82, 'Recall': 0.79, 'F1': 0.81}
    },
    'XGBoost': {
        'Accuracy': 0.82,
        '0': {'Precision': 0.83, 'Recall': 0.80, 'F1': 0.82},
        '1': {'Precision': 0.81, 'Recall': 0.84, 'F1': 0.83}
    }
}

# Data Preparation
print("Load Dataset")
file_path = 'Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv'

# Display initial dataset information
print("--- Check Dataset ---")
df = pd.read_csv(file_path)
df.info()

# Select 7 features based on Feature Importance according to the reference paper
paper_features = ['Contract', 'InternetService', 'TotalCharges', 'tenure', 
                'PaperlessBilling', 'MultipleLines', 'StreamingMovies']
X = df[paper_features].copy()
y = df['Churn'].copy()

# Cleaning
X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')
X['TotalCharges'].fillna(X['TotalCharges'].median(), inplace=True)

# Encoding
categorical_columns = ['Contract', 'InternetService', 'PaperlessBilling', 'MultipleLines', 'StreamingMovies']
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
y = LabelEncoder().fit_transform(y)

# Scaling
scaler = StandardScaler()
X[['tenure', 'TotalCharges']] = scaler.fit_transform(X[['tenure', 'TotalCharges']])

# SMOTE
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.2, random_state=42)

# Modelling
print("Training Model\n")
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, eval_metric='logloss')
}

# for print comparison
def print_comparison(model_name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1])
    
    ref = paper_metrics[model_name]

    print(f" METRIC COMPARISON: {model_name.upper()}")

    # Comparasion Accuraxxy
    diff_acc = acc - ref['Accuracy']
    print(f"OVERALL ACCURACY")
    print(f"Script : {acc:.2f} | Paper : {ref['Accuracy']:.2f} | Difference : {diff_acc:+.2f}")
    print("-" * 50)
    
    # Comparison per Class
    for i, class_name in enumerate(['No Churn (0)', 'Churn (1)']):
        print(f"CLASS: {class_name}")
        c_str = str(i)
        
        # Precision
        diff_p = precision[i] - ref[c_str]['Precision']
        print(f"  Precision : Script = {precision[i]:.2f} | Paper = {ref[c_str]['Precision']:.2f} | Difference = {diff_p:+.2f}")
        
        # Recall
        diff_r = recall[i] - ref[c_str]['Recall']
        print(f"  Recall    : Script = {recall[i]:.2f} | Paper = {ref[c_str]['Recall']:.2f} | Difference = {diff_r:+.2f}")
        
        # F1-Score
        diff_f1 = f1[i] - ref[c_str]['F1']
        print(f"  F1-Score  : Script = {f1[i]:.2f} | Paper = {ref[c_str]['F1']:.2f} | Difference = {diff_f1:+.2f}")
        print("-" * 50)
    print("\n")

# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print_comparison(name, y_test, y_pred)