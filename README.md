# 📊 Telco Customer Churn Prediction

## 1. Project Overview
This project implements an end-to-end Machine Learning workflow to predict telecom customer churn. Driven by a robust preprocessing and evaluation framework, the pipeline addresses extreme class imbalance using **SMOTE** and optimizes predictive performance across two powerful classifiers: **Random Forest** and **XGBoost**.

---

## 2. Core Workflow
To eliminate data leakage entirely, all resampling and transformations are encapsulated within a strict cross-validation pipeline. The execution order is as follows:

1. **System Initialization & Data Ingestion:** Loads the raw dataset from `Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
2. **Feature Engineering & Cleaning:** * Trims the feature space to key drivers: `Contract`, `InternetService`, `TotalCharges`, `tenure`, `PaperlessBilling`, `MultipleLines`, `StreamingMovies`.
   * Enforces numeric parsing on `TotalCharges` and handles missing values via median imputation.
3. **Stratified Data Splitting:** Maps target variables (`No` = 0, `Yes` = 1) and partitions the data into an 80:20 Train/Test split, preserving class proportions.
4. **Hyperparameter Tuning:** Executes a 5-fold `StratifiedKFold` cross-validation strategy via `RandomizedSearchCV` to find optimal estimator parameters.
5. **Validation Diagnostics:** Renders learning curves to track training vs. validation accuracy scores across sample sizes, automatically detecting underfitting or overfitting trends.
6. **Explicit Final Training & Evaluation:** Clones the absolute best-tuned estimator configuration, fits it on 100% of the SMOTE-resampled training data, and evaluates final performance against unseen test data using comprehensive classification reports.

---

## 3. Pipeline Architecture
The workflow relies heavily on `imblearn.pipeline.Pipeline` to guarantee that data transformations are isolated per cross-validation fold. The architecture consists of three main stages:

* **Stage 1: Preprocessor (`ColumnTransformer`)**
  * **Numeric Features:** `tenure` and `TotalCharges` are scaled using `StandardScaler`.
  * **Categorical Features:** Remaining features are transformed into binary matrices using `OneHotEncoder`.
* **Stage 2: Resampling (`SMOTE`)**
  * Addresses target class imbalance by synthesizing new data points for the minority class. **Note:** This is strictly applied *only* to the training sets.
* **Stage 3: Classifier**
  * The predictive engine, utilizing either `RandomForestClassifier` or `XGBClassifier`.

---

## 4. Local Execution Guide

### Prerequisites
Ensure your local environment runs Python 3.8+. Install the required dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

### Project Structure

To ensure smooth execution, verify that your workspace directory is structured as follows before triggering the script. The dataset must be located in a dedicated `Dataset` folder relative to the execution script:

```text
📁 Telco_Churn_Project/
│
├── 📁 Dataset/
│   └── 📄 WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw target dataset
│
└── 📄 main.py                                    # Core execution pipeline
```

### Running the Project
Once your environment is set up and the dependencies are installed, navigate to your project root folder within your terminal and execute the pipeline:

```bash
cd path/to/Telco_Churn_Project
python Customer Telco Churn Prediction.py
```

### Future Developmen

To elevate this project from a backend script to a fully functional product, the following developments are planned:

1. **UI/UX Integration:** Developing an interactive web frontend using Streamlit to allow users to input customer parameters and receive real-time churn predictions.
2. **Feature Engineering Expansion:** Exploring and combining existing columns (e.g., creating AverageMonthlyCost from TotalCharges and tenure) to push the F1-Score boundary further.
3. **Advanced Optimization:** Migrating from RandomizedSearchCV to Bayesian Optimization frameworks like Optuna for faster and more precise hyperparameter tuning.
