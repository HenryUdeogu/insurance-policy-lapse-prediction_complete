# 🚗 Auto Insurance Policy Lapse Risk Prediction
This machine learning project predicts whether an insurance policy will lapse (i.e., be canceled or terminated) using real-world features from motor policy data, including claim types and customer behavior.

## 📌 Objective
To proactively identify customers at high risk of lapsing their auto insurance policy, enabling early intervention strategies such as personalized outreach, incentives, or policy adjustments.

---

## 📂 Datasets
- `main_dataset.csv`: Contains customer profiles, contract information, behavioral features, and the target variable `Lapse`.
- `sample_type_claim.csv`: Partial data with detailed claim types and associated costs.

---

## ⚙️ Workflow

### 1. Preprocessing
- Missing value imputation using **kNN**
- Feature extraction from dates: `Customer_age`, `Tenure_years`
- One-hot encoding of categorical variables
- Skew reduction via **log** transformations

### 2. Data Enrichment
- Merged detailed claims data by `Policy_ID`
- Created aggregated and binary features like `HighSeverityClaimCost`, `Has_serious_claim`, etc.

### 3. Class Balancing
- Applied **SMOTE**, **Undersampling**, and **SVM-SMOTE**

### 4. Feature Engineering & Selection
- Used **Variance Inflation Factor (VIF)** for selection
- Kept certain high-VIF features based on SHAP importance

### 5. Modeling
- Evaluated: Logistic Regression, Random Forest, XGBoost
- Hyperparameter tuning with **GridSearchCV**
- Decision threshold tuning for optimal F1-score

---

## 📈 Results

| Metric     | Score  |
|------------|--------|
| **F1-score** (Test) | 0.52 |
| **ROC AUC**         | 0.80 |
| **PR AUC**          | 0.57 |

The final model — **XGBoost trained on undersampled dataset** — achieved the best performance on the test set.

---

## 🧠 Model Explainability
- Used **SHAP (TreeExplainer)** for local and global interpretability
- Compared **SHAP values** with XGBoost's built-in **feature importances**
- Key predictors: `'ClaimsFrequency'`, `'PremiumAmount'`, `Tenure_years`, and `'Total_Cost_claims_year'`


---

## 📊 Visuals
- ROC and PR curves with annotated thresholds
- F1-score vs. threshold plots
- Confusion matrix
- Class imbalance visualization before and after SMOTE

---

## 📦 Requirements

Install dependencies from:

```bash
pip install -r requirements.txt
