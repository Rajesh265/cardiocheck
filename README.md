# Cardiovascular Disease Prediction System

## Problem Statement

This project implements a binary classification system to predict the presence of cardiovascular disease in patients based on various medical and lifestyle attributes. The goal is to build and compare multiple machine learning models to identify the most effective approach for cardiovascular disease prediction.

## Results Summary

After training and evaluating 6 machine learning models on 70,000 patient records:

- **Best Model:** XGBoost with 73.26% accuracy and 0.7957 AUC
- **Total Models Evaluated:** 6 (Logistic Regression, Decision Tree, KNN, Naive Bayes, Random Forest, XGBoost)
- **Total Metrics Calculated:** 6 per model (Accuracy, AUC, Precision, Recall, F1, MCC)
- **Dataset Split:** 80% training (56,000 instances), 20% testing (14,000 instances)

## Dataset Description

**Dataset Name:** Cardiovascular Disease Dataset  
**Source:** Kaggle - Cardiovascular Disease Dataset  
**Type:** Binary Classification (Disease Present: 1, No Disease: 0)

### Dataset Specifications:
- **Number of Instances:** 70,000
- **Number of Features:** 12 (excluding target)
- **Target Variable:** Binary (0 = No Disease, 1 = Disease Present)

### Feature Descriptions:

| Feature | Description | Type | Values |
|---------|-------------|------|--------|
| **age** | Age in years (converted from days) | Integer | 30-65 years |
| **gender** | Gender | Binary | 1 = female, 2 = male |
| **height** | Height in cm | Integer | 140-200 cm |
| **weight** | Weight in kg | Float | 40-150 kg |
| **ap_hi** | Systolic blood pressure | Integer | 80-200 mm Hg |
| **ap_lo** | Diastolic blood pressure | Integer | 50-120 mm Hg |
| **cholesterol** | Cholesterol level | Categorical | 1: normal, 2: above normal, 3: well above normal |
| **gluc** | Glucose level | Categorical | 1: normal, 2: above normal, 3: well above normal |
| **smoke** | Smoking status | Binary | 0 = no, 1 = yes |
| **alco** | Alcohol intake | Binary | 0 = no, 1 = yes |
| **active** | Physical activity | Binary | 0 = no, 1 = yes |
| **cardio** | Cardiovascular disease presence (TARGET) | Binary | 0 = no, 1 = yes |

### Dataset Characteristics:
- **Class Distribution:** Balanced dataset (approximately 50-50 split)
- **Missing Values:** No missing values
- **Feature Types:** Mix of continuous and categorical features
- **Clinical Relevance:** Standard health measurements and lifestyle factors

---

## Models Used

Six classification models will be implemented and evaluated on the Cardiovascular Disease dataset:

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 71.34% | 0.7780 | 73.12% | 67.45% | 70.17% | 0.4281 |
| Decision Tree | 62.61% | 0.6283 | 62.90% | 61.35% | 62.12% | 0.2522 |
| K-Nearest Neighbors | 64.42% | 0.6851 | 64.84% | 62.94% | 63.87% | 0.2885 |
| Naive Bayes | 58.94% | 0.6850 | 71.33% | 29.83% | 42.07% | 0.2195 |
| Random Forest | 70.65% | 0.7605 | 70.96% | 69.85% | 70.40% | 0.4130 |
| XGBoost | **73.26%** | **0.7957** | **75.05%** | 69.63% | **72.24%** | **0.4663** |

**Best Overall Model:** XGBoost (highest accuracy, AUC, precision, F1, and MCC)

---

## Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | Moderate performance with 71.34% accuracy. High precision (73.12%) indicates fewer false positives. MCC of 0.428 shows moderate correlation. |
| Decision Tree | Moderate performance with 62.61% accuracy. High precision (62.90%) indicates fewer false positives. MCC of 0.252 shows moderate correlation. |
| K-Nearest Neighbors | Moderate performance with 64.42% accuracy. High precision (64.84%) indicates fewer false positives. MCC of 0.288 shows moderate correlation. |
| Naive Bayes | Moderate performance with 58.94% accuracy. High precision (71.33%) indicates fewer false positives. MCC of 0.220 shows moderate correlation. |
| Random Forest | Moderate performance with 70.65% accuracy. High precision (70.96%) indicates fewer false positives. MCC of 0.413 shows moderate correlation. |
| XGBoost | Moderate performance with 73.26% accuracy. High precision (75.05%) indicates fewer false positives. MCC of 0.466 shows moderate correlation. |

### Key Findings:
- **XGBoost** achieved the best overall performance across most metrics (73.26% accuracy, 0.7957 AUC)
- **Random Forest** had the highest recall (69.85%), making it best at identifying positive cases
- **Naive Bayes** showed the lowest performance (58.94% accuracy) with very low recall (29.83%)
- All models demonstrate moderate correlation (MCC between 0.22-0.47) indicating room for improvement
- Precision is consistently higher than recall across models, suggesting they are conservative in predicting disease

---

## How to Run

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository:**
```bash
git clone <your-github-repo-url>
cd heart-disease-prediction
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Train models:**
   - Open and run all cells in `model/train_models.ipynb`
   - This will train all 6 models and generate metrics

4. **Run Streamlit app:**
```bash
python3 -m streamlit run streamlit_app.py
# or simply:
streamlit run streamlit_app.py
```

5. **Access the app:**
   - Open your browser and navigate to the Local URL shown (typically http://localhost:8501 or 8502)
   - Upload the test data CSV file
   - Select a model and view predictions

---

## Project Structure

```
heart-disease-prediction/
│
├── streamlit_app.py                # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── model/                          # Model files and training
│   ├── train_models.ipynb         # Training notebook
│   ├── *.pkl                      # Trained models (6 models)
│   ├── scaler.pkl                 # Feature scaler
│   ├── metrics_comparison.csv     # Metrics table
│   └── observations.csv           # Performance observations
│
└── data/                          
    ├── cardio_train.csv           # Full dataset (70,000 instances)
    └── test_data.csv              # Test data for app
```

---

## Deployment

Push to GitHub and deploy on Streamlit Community Cloud.

---

## Features

**Streamlit Application:**
- CSV file upload for test data
- Model selection dropdown (6 models)
- Evaluation metrics display (Accuracy, AUC, Precision, Recall, F1, MCC)
- Confusion matrix visualization
- Classification report
- Model comparison
