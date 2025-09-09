# 💳 Fraud Detection with Machine Learning

This project focuses on **detecting fraudulent financial transactions** using Machine Learning.  
The dataset contains **6M+ records** and highlights the challenge of **class imbalance** — where fraudulent cases are very rare compared to normal transactions.  

---

## 📊 Dataset Overview
- **Source**: Simulated financial transactions dataset  
- **Rows**: ~6,362,620  
- **Columns**: step, type, amount, balances, fraud labels  

### Key Features
- `step`: Time unit (1 step = 1 hour, 744 steps = 30 days)  
- `type`: Transaction type (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER)  
- `amount`: Transaction amount  
- `oldbalanceOrg`, `newbalanceOrig`: Sender balances  
- `oldbalanceDest`, `newbalanceDest`: Recipient balances  
- `isFraud`: Fraudulent transaction flag (target)  
- `isFlaggedFraud`: Rule-based flag (amount > 200,000)  

---

## ⚙️ Workflow
1. **Data Cleaning & Preprocessing**
   - Removed unnecessary columns (`nameOrig`, `nameDest`)  
   - One-hot encoded transaction `type`  
   - Created new engineered features:  
     - `errorBalanceOrig` = oldbalanceOrg – amount – newbalanceOrig  
     - `errorBalanceDest` = newbalanceDest – oldbalanceDest – amount  

2. **Feature Scaling**  
   - StandardScaler applied for normalization  

3. **Model Development**  
   - Used **XGBoost Classifier** with:
     - `scale_pos_weight` to handle imbalance  
     - Hyperparameter tuning (n_estimators, max_depth, learning_rate, etc.)  

4. **Evaluation Metrics**  
   - Classification Report (Precision, Recall, F1-score)  
   - Confusion Matrix  
   - **ROC-AUC Curve**  

---

## 📈 Results
- **ROC-AUC Score**: ~0.9997  
- **Precision (Fraud Class)**: 0.88  
- **Recall (Fraud Class)**: 0.99  
- Fraud transactions are mostly detected in **TRANSFER** and **CASH_OUT** types.  
- Balance mismatches (`errorBalanceOrig`, `errorBalanceDest`) are strong predictors.  

---

## 🔑 Key Insights
- Fraudulent activity is concentrated in **high-value TRANSFER and CASH_OUT** operations.  
- **Balance inconsistencies** are strong fraud indicators.  
- Real-time monitoring of these features can reduce financial fraud significantly.  

---

## 🛡️ Business Recommendations
- Implement **real-time alerts** for balance mismatches.  
- Flag high-value TRANSFER/CASH_OUT for extra verification.  
- Use adaptive thresholds depending on customer behavior.  
- Continuously retrain the model to adapt to new fraud patterns.  

---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  
- Matplotlib / Seaborn  

---
