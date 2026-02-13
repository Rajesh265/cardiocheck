# Heart Disease Prediction System

## 🎯 Problem Statement

This project implements a **binary classification system** to predict the presence of heart disease in patients based on various medical attributes. The goal is to build and compare multiple machine learning models to identify the most effective approach for heart disease prediction, enabling early diagnosis and potentially life-saving interventions.

## 📊 Dataset Description

**Dataset Name:** UCI Heart Disease Dataset  
**Source:** [UCI Machine Learning Repository - Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+disease)  
**Type:** Binary Classification (Disease Present: 1, No Disease: 0)

### Dataset Specifications:
- **Number of Instances:** 303 (Cleveland dataset) - *Note: Extended dataset with 920+ instances can be used by combining Cleveland, Hungary, Switzerland, and VA Long Beach datasets*
- **Number of Features:** 13
- **Target Variable:** Binary (0 = No Disease, 1 = Disease Present)

### Feature Descriptions:

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| **age** | Age in years | Integer | 29-77 |
| **sex** | Sex (1 = male, 0 = female) | Binary | 0, 1 |
| **cp** | Chest pain type | Categorical | 0-3 |
|  | • 0: Typical angina | | |
|  | • 1: Atypical angina | | |
|  | • 2: Non-anginal pain | | |
|  | • 3: Asymptomatic | | |
| **trestbps** | Resting blood pressure (mm Hg) | Integer | 94-200 |
| **chol** | Serum cholesterol (mg/dl) | Integer | 126-564 |
| **fbs** | Fasting blood sugar > 120 mg/dl | Binary | 0, 1 |
| **restecg** | Resting electrocardiographic results | Categorical | 0-2 |
|  | • 0: Normal | | |
|  | • 1: ST-T wave abnormality | | |
|  | • 2: Left ventricular hypertrophy | | |
| **thalach** | Maximum heart rate achieved | Integer | 71-202 |
| **exang** | Exercise induced angina | Binary | 0, 1 |
| **oldpeak** | ST depression induced by exercise | Float | 0-6.2 |
| **slope** | Slope of peak exercise ST segment | Categorical | 0-2 |
|  | • 0: Upsloping | | |
|  | • 1: Flat | | |
|  | • 2: Downsloping | | |
| **ca** | Number of major vessels colored by fluoroscopy | Integer | 0-3 |
| **thal** | Thalassemia | Categorical | 1-3 |
|  | • 1: Normal | | |
|  | • 2: Fixed defect | | |
|  | • 3: Reversible defect | | |
| **target** | Disease presence | Binary | 0, 1 |

### Dataset Characteristics:
- **Class Distribution:** Relatively balanced between disease and no disease cases
- **Missing Values:** Some features contain missing values (handled during preprocessing)
- **Feature Types:** Mix of continuous and categorical features
- **Clinical Relevance:** All features are standard medical measurements used in cardiovascular health assessment

---

## 🤖 Models Used

Six classification models were implemented and evaluated on the Heart Disease dataset:

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| Logistic Regression | 0.8333 | 0.9498 | 0.8462 | 0.7857 | 0.8148 | 0.6652 |
| Decision Tree | 0.7000 | 0.6964 | 0.6923 | 0.6429 | 0.6667 | 0.3955 |
| K-Nearest Neighbors (kNN) | 0.8833 | 0.9492 | 0.9200 | 0.8214 | 0.8679 | 0.7680 |
| Naive Bayes | 0.8833 | 0.9375 | 0.8889 | 0.8571 | 0.8727 | 0.7655 |
| Random Forest (Ensemble) | 0.8500 | 0.9414 | 0.8800 | 0.7857 | 0.8302 | 0.7002 |
| XGBoost (Ensemble) | 0.8667 | 0.8917 | 0.8846 | 0.8214 | 0.8519 | 0.7326 |

**Key Findings:**
- **Best Overall:** K-Nearest Neighbors and Naive Bayes (88.33% accuracy)
- **Best AUC:** Logistic Regression (0.9498) - excellent at distinguishing classes
- **Best Precision:** K-Nearest Neighbors (0.9200) - fewest false positives
- **Best F1 & Recall:** Naive Bayes (0.8727 F1, 0.8571 recall)
- **Lowest Performance:** Decision Tree (70% accuracy) - shows signs of overfitting

---

## 📈 Model Performance Observations

### Observations Table

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Good performance with 83.33% accuracy. High precision (84.62%) indicates fewer false positives. MCC of 0.665 shows strong correlation. Excellent AUC (0.9498) demonstrates superior ability to distinguish between classes. Well-suited for interpretable binary classification. |
| **Decision Tree** | Moderate performance with 70.00% accuracy. Shows signs of overfitting with lower generalization. Precision of 69.23% and recall of 64.29% indicate room for improvement. MCC of 0.396 suggests moderate predictive power. May benefit from pruning or ensemble methods. |
| **K-Nearest Neighbors (kNN)** | Very Good performance with 88.33% accuracy - tied for best overall. Outstanding precision (92.00%) minimizes false positives. Strong MCC (0.768) indicates excellent predictive capability. Feature scaling was crucial for optimal performance. Sensitive to distance metric and k value selection. |
| **Naive Bayes** | Very Good performance with 88.33% accuracy - tied for best overall. Best F1 score (0.8727) shows excellent balance. High recall (85.71%) effectively captures most disease cases. Despite independence assumption violations, performs remarkably well on this dataset. Fast training and prediction make it practical for deployment. |
| **Random Forest (Ensemble)** | Good performance with 85.00% accuracy. High precision (88.00%) reduces false alarms. MCC of 0.700 shows strong correlation. Benefits from ensemble aggregation reducing overfitting compared to single Decision Tree. Feature importance insights valuable for clinical interpretation. |
| **XGBoost (Ensemble)** | Very Good performance with 86.67% accuracy. Well-balanced precision (88.46%) and recall (82.14%). Strong MCC (0.733) demonstrates robust predictive power. Gradient boosting provides iterative improvement. Computational efficiency and regularization prevent overfitting. |

---

## 🚀 How to Run

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository:**
```bash
git clone <your-github-repo-url>
cd heart-disease-prediction
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv

# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Training Models

4. **Run the Jupyter notebook to train models:**
```bash
jupyter notebook model/train_models.ipynb
```
- Execute all cells in the notebook
- This will download the dataset, train all 6 models, calculate metrics, and save:
  - Trained models in `model/` folder
  - Test data in `data/test_data.csv`
  - Metrics in `model/metrics_comparison.csv`

### Running the Streamlit App

5. **Launch the Streamlit application:**
```bash
streamlit run app.py
```

6. **Use the application:**
- Open your browser (usually auto-opens to `http://localhost:8501`)
- Upload the test data CSV from `data/test_data.csv`
- Select a model from the dropdown
- View predictions, metrics, confusion matrix, and classification report

---

## 📁 Project Structure

```
heart-disease-prediction/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
│
├── model/                          # Model files and training code
│   ├── train_models.ipynb         # Jupyter notebook for model training
│   ├── logistic_regression.pkl    # Trained Logistic Regression model
│   ├── decision_tree.pkl          # Trained Decision Tree model
│   ├── k_nearest_neighbors.pkl    # Trained KNN model
│   ├── naive_bayes.pkl            # Trained Naive Bayes model
│   ├── random_forest.pkl          # Trained Random Forest model
│   ├── xgboost.pkl                # Trained XGBoost model
│   ├── scaler.pkl                 # StandardScaler for feature scaling
│   ├── metrics_comparison.csv     # Model metrics comparison
│   └── observations.csv           # Model performance observations
│
└── data/                          # Dataset files
    ├── heart_disease.csv          # Raw dataset (downloaded)
    └── test_data.csv              # Test dataset for Streamlit app
```

---

## 🌐 Deployment

**Live Application:** *[Will be deployed on Streamlit Community Cloud]*

### Deployment Steps:
1. Push code to GitHub repository
2. Sign in to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Create new app pointing to this repository
4. Select `app.py` as the main file
5. Deploy and share the live link

---

## 📊 Features

### Streamlit Web Application Features:
- ✅ **CSV File Upload:** Upload test data for predictions
- ✅ **Model Selection:** Choose from 6 different classification models
- ✅ **Evaluation Metrics:** Display all 6 metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- ✅ **Confusion Matrix:** Visual representation of prediction results
- ✅ **Classification Report:** Detailed performance breakdown
- ✅ **Interactive UI:** User-friendly interface with clear instructions
