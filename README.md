# Project Documentation: Telco Customer Churn Prediction

## Project Description
This project aims to predict the probability of customers canceling their subscription (churn). By analyzing historical customer behavior using the Telco Customer Churn dataset, the system is designed to provide targeted, strategic recommendations for the marketing team.

## Data Pre-processing Pipeline
* Data cleaning was performed by dropping identifier columns and imputing missing numerical values using the median.
* Categorical features with logical tiers, such as contract duration, were transformed using **Ordinal Encoding**.
* **One-Hot Encoding** was applied to the remaining non-ordered categorical variables.
* The dataset was split into **80% training data** and **20% testing data**.
* Feature scaling (**Min-Max Scaling**) and class balancing (**SMOTE**) were implemented exclusively on the training data to prevent data leakage.

## Model Experiment Status & Main Issues
* **Data Leakage Bug:** The current evaluation metrics are invalid because the models generated predictions using unscaled test data (`X_test` instead of `X_test_scaled`).
* **Logistic Regression:** This model exhibits over-sensitivity, taking a shortcut by predicting almost all customers as churn, which results in naive predictions.
* **Decision Tree:** The current performance comparison is not apple-to-apple because this algorithm was used with default parameters and has not undergone hyperparameter tuning.
* **Random Forest:** This algorithm shows promising class separation capabilities (AUC-ROC 0.79), but fails to classify the minority class effectively due to Scikit-Learn's rigid default probability threshold (0.5).

## Next Steps (Action Plan)
- [ ] Fix the data leakage bug by ensuring all model predictions use the scaled test data (`X_test_scaled`).
- [ ] Implement `GridSearchCV` on the Decision Tree model to ensure objective and fair evaluation.
- [ ] Perform threshold tuning on the Random Forest model using probabilities to make it more aggressive and responsive in detecting churn signals.
- [ ] Integrate the entire workflow into a unified `sklearn.pipeline.Pipeline` to permanently prevent data leakage.
- [ ] Build a real-time prediction user interface using **Streamlit**.

---
*Documented for Machine Learning Project Progress.*