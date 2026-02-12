# Customer-churn-prediction-model-

# 📊 Customer Churn Prediction Project

## 🔹 Overview
This project predicts whether a customer will churn (leave a service) using machine learning models. It includes full data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and saved artifacts for deployment.

---

## 🎯 Objective
To build a classification model that accurately predicts customer churn using customer usage behavior, service details, and account information.

---

## 📁 Dataset Information
**File:** `customer_churn_dataset.csv`  
**Rows:** 20,000  
**Columns:** 11  

### Features
| Feature | Description |
|-------|-------------|
customer_id | Unique ID |
tenure | Months customer stayed |
monthly_charges | Monthly bill |
total_charge | Total paid |
contract | Contract type |
payment_method | Payment mode |
internet_service | Internet type |
tech_support | Support subscription |
online_security | Security service |
support_calls | Customer support calls |
churn | Target variable |

---

## ⚙️ Project Pipeline

### 1️⃣ Data Cleaning
- Removed missing values
- Dropped irrelevant columns (like `customer_id`)

### 2️⃣ Encoding
Categorical columns converted using **LabelEncoder**

Saved encoders:
```
encoders.pkl
label_encoders.pkl
```

### 3️⃣ Exploratory Data Analysis (EDA)
Performed:
- Distribution plots
- Boxplots
- Correlation heatmap
- Class distribution analysis

### 4️⃣ Handling Imbalanced Data
Applied **SMOTE** to balance churn classes.

### 5️⃣ Models Used
- Decision Tree Classifier
- Random Forest Classifier

Cross-validation (5-fold) used for evaluation.

### 6️⃣ Model Evaluation
Metrics used:
- Accuracy
- Confusion Matrix
- Classification Report

Visualizations:
- Confusion matrix heatmap
- Feature importance graph

---

## 🧠 Saved Artifacts
| File | Purpose |
|----|--------|
feature_names.pkl | Stores feature order |
encoders.pkl | Encoding mappings |
label_encoders.pkl | Column encoders |

These allow direct prediction without retraining.

---

## 🚀 How to Run Project

### Install dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost
```

### Run notebook
```bash
jupyter notebook "CUSTOMER CHURN PROJECT.ipynb"
```

---

## 🔮 Predict Using Saved Model (Example)
```python
import pickle

with open("model.pkl","rb") as f:
    model = pickle.load(f)

prediction = model.predict([[12, 70.5, 845.4, 1,2,0,1,0,3]])
print(prediction)
```

---

## 📈 Key Insights
- Customers with month-to-month contracts churn more.
- Higher support calls correlate with churn.
- Long-tenure customers are less likely to churn.

---

## 🏆 Skills Demonstrated
✔ Data Cleaning  
✔ Feature Engineering  
✔ Visualization  
✔ Handling Imbalanced Data  
✔ Model Evaluation  
✔ Serialization (Pickle)  
✔ ML Pipeline Design  

---

## 📌 Future Improvements
- Hyperparameter tuning
- Deployment via Flask / Streamlit
- Feature scaling optimization
- Model comparison with boosting algorithms

---

## 👨‍💻 Author
**Siddharth Purohit**  
Aspiring Data Scientist | Machine Learning Enthusiast

