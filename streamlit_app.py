"""
Cardiovascular Disease Prediction - Streamlit Web Application
Features: Model selection, metrics display, confusion matrix, dataset upload
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Cardiovascular Disease Prediction",
    layout="wide"
)

# Custom CSS for clean blue and red theme
st.markdown("""
    <style>
    /* Main background */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8f0f7 100%);
    }
    
    /* Main header */
    .main-header {
        font-size: 15rem;
        font-weight: 800;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3);
        letter-spacing: -0.5px;
        line-height: 1.1;
    }
    
    /* Sub-header */
    .sub-header {
        font-size: 15rem;
        color: white;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 600;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
        letter-spacing: 0.5px;
        line-height: 1.2;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e40af 0%, #1e3a8a 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: white;
    }
    
    /* Info boxes */
    .info-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border-left: 4px solid #3b82f6;
        color: #1e293b;
    }
    
    .success-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border-left: 4px solid #10b981;
        color: #1e293b;
    }
    
    .warning-box {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        border-left: 4px solid #ef4444;
        color: #1e293b;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(37, 99, 235, 0.3);
    }
    
    /* Metrics display */
    [data-testid="stMetricValue"] {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
    }
    
    [data-testid="stMetricLabel"] {
        color: white;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    /* Info box text */
    .info-box {
        font-size: 1.1rem;
    }
    
    .info-box h3 {
        font-size: 1.5rem;
    }
    
    .success-box {
        font-size: 1.1rem;
    }
    
    .warning-box {
        font-size: 1.1rem;
    }
    
    /* Upload area */
    [data-testid="stFileUploader"] {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        border: 2px dashed #3b82f6;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.6rem 1rem !important;
        box-shadow: 0 4px 6px rgba(239, 68, 68, 0.2) !important;
        transition: all 0.3s ease !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 12px rgba(239, 68, 68, 0.3) !important;
    }
    
    /* Selectbox */
    [data-baseweb="select"] {
        background: white;
        border-radius: 8px;
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #cbd5e1, transparent);
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.title("Cardiovascular Disease Prediction System")

st.markdown("""
<div class="info-box">
### How to Use
1. Upload your test dataset in CSV format
2. Select a classification model from the dropdown
3. View evaluation metrics, confusion matrix, and classification report

**Dataset Requirements:** CSV must contain 12 features: age, gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, and cardio
</div>
""", unsafe_allow_html=True)

# Sidebar for model selection and file upload
st.sidebar.header("Configuration")

# Model paths
model_options = {
    'Logistic Regression': 'model/logistic_regression.pkl',
    'Decision Tree': 'model/decision_tree.pkl',
    'K-Nearest Neighbors': 'model/k_nearest_neighbors.pkl',
    'Naive Bayes': 'model/naive_bayes.pkl',
    'Random Forest': 'model/random_forest.pkl',
    'XGBoost': 'model/xgboost.pkl'
}

# Model Selection Section
st.sidebar.subheader("Select Model")

selected_model_name = st.sidebar.selectbox(
    "Model Selection",
    list(model_options.keys()),
    help="Choose which model to use for predictions",
    label_visibility="collapsed"
)

# File Upload Section
st.sidebar.subheader("Upload Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV",
    type=['csv'],
    help="Upload a CSV file containing test data with features and cardio column",
    label_visibility="collapsed"
)

# Sample data download button
st.sidebar.markdown("""
<p style="color: white; font-size: 0.95rem; margin: 1rem 0 0.5rem 0; font-weight: 500;">
    Need sample data?
</p>
""", unsafe_allow_html=True)

# Load sample data for download
try:
    with open('data/test_data.csv', 'rb') as f:
        sample_data = f.read()
    
    st.sidebar.download_button(
        label="Download Sample Test Data",
        data=sample_data,
        file_name="heart_disease_sample.csv",
        mime="text/csv",
        help="Download a sample test dataset to try the application",
        use_container_width=True
    )
except FileNotFoundError:
    st.sidebar.info("Sample data not available")

# About Section
st.sidebar.markdown("### About")
st.sidebar.markdown("ML classification system with 6 algorithms trained on Cardiovascular Disease dataset.")
st.sidebar.markdown("""
**6 Models Available:**
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors
- Naive Bayes
- Random Forest
- XGBoost
""")

# Main content
if uploaded_file is not None:
    try:
        # Load the uploaded data
        df = pd.read_csv(uploaded_file)
        
        st.markdown(f"""
        <div class="success-box">
            <strong style="font-size: 1.1rem;">File Uploaded Successfully!</strong><br>
            <span style="font-size: 1.05rem;">Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Display first few rows
        with st.expander("View Uploaded Data (First 10 rows)", expanded=False):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Check if cardio column exists
        if 'cardio' not in df.columns:
            st.markdown("""
            <div class="warning-box">
                <strong style="font-size: 1.1rem;">Error:</strong> <span style="font-size: 1.05rem;">'cardio' column not found in the uploaded CSV file!</span>
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        
        # Separate features and target
        X_test = df.drop('cardio', axis=1)
        y_test = df['cardio']
        
        # Validate number of features
        expected_features = 11
        if X_test.shape[1] != expected_features:
            st.markdown(f"""
            <div class="warning-box">
                <strong style="font-size: 1.1rem;">Warning:</strong> <span style="font-size: 1.05rem;">Expected {expected_features} features, but found {X_test.shape[1]}</span>
            </div>
            """, unsafe_allow_html=True)
        
        # Load the selected model
        model_path = model_options[selected_model_name]
        
        if not os.path.exists(model_path):
            st.markdown(f"""
            <div class="warning-box">
                <strong style="font-size: 1.1rem;">Model Not Found:</strong> {model_path}<br>
                <span style="font-size: 1.05rem;">Please ensure models are trained first by running the Jupyter notebook.</span>
            </div>
            """, unsafe_allow_html=True)
            st.stop()
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        # Load scaler if needed (for Logistic Regression and KNN)
        if selected_model_name in ['Logistic Regression', 'K-Nearest Neighbors']:
            scaler_path = 'model/scaler.pkl'
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                X_test_processed = scaler.transform(X_test)
            else:
                st.warning("Scaler not found. Using unscaled data.")
                X_test_processed = X_test
        else:
            X_test_processed = X_test
        
        # Make predictions
        y_pred = model.predict(X_test_processed)
        y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
        
        st.markdown(f"""
        <div class="success-box">
            <strong style="font-size: 1.1rem;">Predictions Complete!</strong><br>
            <span style="font-size: 1.05rem;">Using <strong>{selected_model_name}</strong> model</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Calculate metrics
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0 1rem 0;">
            <h2 style="color: white; 
                       font-size: 2.2rem; font-weight: 700;">
                Evaluation Metrics
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        metrics = {
            'Accuracy': accuracy_score(y_test, y_pred),
            'AUC Score': roc_auc_score(y_test, y_pred_proba),
            'Precision': precision_score(y_test, y_pred, zero_division=0),
            'Recall': recall_score(y_test, y_pred, zero_division=0),
            'F1 Score': f1_score(y_test, y_pred, zero_division=0),
            'MCC': matthews_corrcoef(y_test, y_pred)
        }
        
        # Display metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            st.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
        
        with col2:
            st.metric("Precision", f"{metrics['Precision']:.4f}")
            st.metric("Recall", f"{metrics['Recall']:.4f}")
        
        with col3:
            st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
            st.metric("MCC", f"{metrics['MCC']:.4f}")
        
        # Metrics table
        st.markdown("""
        <h3 style="color: white; 
                   font-weight: 600; margin-top: 2rem; font-size: 1.5rem;">
            Detailed Metrics
        </h3>
        """, unsafe_allow_html=True)
        metrics_df = pd.DataFrame([metrics])
        st.dataframe(
            metrics_df.style.format("{:.4f}").background_gradient(cmap='Blues', axis=1),
            use_container_width=True
        )
        
        # Confusion Matrix
        st.markdown("""
        <div style="text-align: center; margin: 3rem 0 1rem 0;">
            <h2 style="color: white; 
                       font-size: 2.2rem; font-weight: 700;">
                Confusion Matrix
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        cm = confusion_matrix(y_test, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Use blue color palette
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease', 'Disease'],
            yticklabels=['No Disease', 'Disease'],
            cbar_kws={'label': 'Count'},
            ax=ax,
            linewidths=2,
            linecolor='#1e40af',
            annot_kws={'size': 16, 'weight': 'bold', 'color': '#1e293b'}
        )
        ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold', color='#1e40af')
        ax.set_ylabel('True Label', fontsize=14, fontweight='bold', color='#1e40af')
        ax.set_title(f'Confusion Matrix - {selected_model_name}', 
                    fontsize=16, fontweight='bold', pad=20, color='#1e40af')
        
        # Set background color to light
        fig.patch.set_facecolor('#f8fafc')
        ax.set_facecolor('#ffffff')
        ax.tick_params(colors='#475569')
        ax.xaxis.label.set_color('#1e40af')
        ax.yaxis.label.set_color('#1e40af')
        ax.title.set_color('#1e40af')
        
        st.pyplot(fig)
        
        # Classification Report
        st.subheader("Classification Report")
        
        # Get classification report as dictionary
        from sklearn.metrics import classification_report
        report_dict = classification_report(y_test, y_pred, target_names=['No Disease', 'Disease'], output_dict=True)
        
        # Create a styled dataframe
        report_df = pd.DataFrame({
            'Class': ['No Disease', 'Disease', 'Macro Avg', 'Weighted Avg'],
            'Precision': [
                report_dict['No Disease']['precision'],
                report_dict['Disease']['precision'],
                report_dict['macro avg']['precision'],
                report_dict['weighted avg']['precision']
            ],
            'Recall': [
                report_dict['No Disease']['recall'],
                report_dict['Disease']['recall'],
                report_dict['macro avg']['recall'],
                report_dict['weighted avg']['recall']
            ],
            'F1-Score': [
                report_dict['No Disease']['f1-score'],
                report_dict['Disease']['f1-score'],
                report_dict['macro avg']['f1-score'],
                report_dict['weighted avg']['f1-score']
            ],
            'Support': [
                int(report_dict['No Disease']['support']),
                int(report_dict['Disease']['support']),
                int(report_dict['macro avg']['support']),
                int(report_dict['weighted avg']['support'])
            ]
        })
        
        # Display as HTML table with custom styling
        st.markdown("""
        <style>
        .report-table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .report-table thead {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
            color: white;
        }
        .report-table th {
            padding: 1rem;
            text-align: center;
            font-weight: 600;
            font-size: 1rem;
        }
        .report-table td {
            padding: 0.9rem;
            text-align: center;
            border-bottom: 1px solid #e5e7eb;
            color: #1e293b;
            font-size: 0.95rem;
        }
        .report-table tbody tr:hover {
            background-color: #f8fafc;
        }
        .report-table .class-col {
            font-weight: 600;
            text-align: left;
            color: #1e40af;
        }
        .report-table .avg-row {
            background-color: #f1f5f9;
            font-weight: 600;
        }
        .report-table .metric-good {
            color: #059669;
            font-weight: 600;
        }
        </style>
        """, unsafe_allow_html=True)
        
        html_table = f"""
        <table class="report-table">
            <thead>
                <tr>
                    <th>Class</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1-Score</th>
                    <th>Support</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="class-col">No Disease</td>
                    <td class="metric-good">{report_dict['No Disease']['precision']:.2f}</td>
                    <td class="metric-good">{report_dict['No Disease']['recall']:.2f}</td>
                    <td class="metric-good">{report_dict['No Disease']['f1-score']:.2f}</td>
                    <td>{int(report_dict['No Disease']['support'])}</td>
                </tr>
                <tr>
                    <td class="class-col">Disease</td>
                    <td class="metric-good">{report_dict['Disease']['precision']:.2f}</td>
                    <td class="metric-good">{report_dict['Disease']['recall']:.2f}</td>
                    <td class="metric-good">{report_dict['Disease']['f1-score']:.2f}</td>
                    <td>{int(report_dict['Disease']['support'])}</td>
                </tr>
                <tr class="avg-row">
                    <td class="class-col">Macro Avg</td>
                    <td>{report_dict['macro avg']['precision']:.2f}</td>
                    <td>{report_dict['macro avg']['recall']:.2f}</td>
                    <td>{report_dict['macro avg']['f1-score']:.2f}</td>
                    <td>{int(report_dict['macro avg']['support'])}</td>
                </tr>
                <tr class="avg-row">
                    <td class="class-col">Weighted Avg</td>
                    <td>{report_dict['weighted avg']['precision']:.2f}</td>
                    <td>{report_dict['weighted avg']['recall']:.2f}</td>
                    <td>{report_dict['weighted avg']['f1-score']:.2f}</td>
                    <td>{int(report_dict['weighted avg']['support'])}</td>
                </tr>
            </tbody>
        </table>
        """
        
        st.markdown(html_table, unsafe_allow_html=True)
        
        # Add metric explanations
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; margin-top: 1.5rem; 
                    border-left: 4px solid #8b5cf6; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);">
            <h4 style="color: #1e293b; margin: 0 0 1rem 0;">📖 Metrics Explained:</h4>
            <div style="color: #475569; line-height: 1.8;">
                <strong style="color: #3b82f6;">Precision:</strong> Of all predicted positives, how many were correct? (Accuracy of positive predictions)<br>
                <strong style="color: #3b82f6;">Recall:</strong> Of all actual positives, how many did we catch? (Coverage of actual cases)<br>
                <strong style="color: #3b82f6;">F1-Score:</strong> Harmonic mean of precision and recall (Balanced performance metric)<br>
                <strong style="color: #3b82f6;">Support:</strong> Number of actual samples in each class
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Additional insights
        st.markdown("""
        <div style="text-align: center; margin: 3rem 0 1rem 0;">
            <h2 style="color: white; 
                       font-size: 2.2rem; font-weight: 700;">
                Prediction Insights
            </h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            <div class="success-box">
                <strong>Correct Predictions</strong><br><br>
                True Positives (Disease Correctly Identified): <strong>{cm[1, 1]}</strong><br>
                True Negatives (No Disease Correctly Identified): <strong>{cm[0, 0]}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="warning-box">
                <strong>Incorrect Predictions</strong><br><br>
                False Positives (Incorrectly Predicted Disease): <strong>{cm[0, 1]}</strong><br>
                False Negatives (Missed Disease Cases): <strong>{cm[1, 0]}</strong>
            </div>
            """, unsafe_allow_html=True)
        
        # Prediction distribution
        st.markdown("""
        <h3 style="color: white; 
                   font-weight: 600; margin-top: 2rem;">
            Prediction Distribution
        </h3>
        """, unsafe_allow_html=True)
        
        pred_counts = pd.Series(y_pred).value_counts()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#3b82f6', '#ef4444']
        bars = pred_counts.plot(kind='bar', color=colors, ax=ax, edgecolor='#1e40af', linewidth=2)
        ax.set_xlabel('Prediction', fontsize=14, fontweight='bold', color='#1e40af')
        ax.set_ylabel('Count', fontsize=14, fontweight='bold', color='#1e40af')
        ax.set_title('Distribution of Predictions', fontsize=16, fontweight='bold', pad=20, color='#1e40af')
        ax.set_xticklabels(['No Disease', 'Disease'], rotation=0, fontsize=12, color='#475569')
        ax.grid(axis='y', alpha=0.2, linestyle='--', color='#cbd5e1')
        ax.set_facecolor('#ffffff')
        fig.patch.set_facecolor('#f8fafc')
        ax.tick_params(colors='#475569')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add value labels on bars
        for i, v in enumerate(pred_counts):
            ax.text(i, v + 0.5, str(v), ha='center', fontweight='bold', fontsize=14, color='#1e293b')
        
        st.pyplot(fig)
        
    except Exception as e:
        st.markdown(f"""
        <div class="warning-box">
            <strong>An Error Occurred:</strong><br>
            {str(e)}<br><br>
            Please ensure your CSV file is properly formatted and contains all required columns.
        </div>
        """, unsafe_allow_html=True)

else:
    # Show placeholder when no file is uploaded
    st.markdown("""
    <div class="info-box" style="margin-top: 3rem;">
        <h3 style="margin-top: 0; color: #1e40af;">Getting Started</h3>
        <p style="font-size: 1.1rem; color: #475569; margin-bottom: 1rem;">
            Ready to analyze heart disease predictions? Follow these steps:
        </p>
        <ol style="font-size: 1rem; color: #334155; line-height: 2;">
            <li><strong style="color: #2563eb;">Download Sample Data:</strong> Click the download button in the sidebar</li>
            <li><strong style="color: #2563eb;">Upload CSV:</strong> Upload the sample file using the file uploader</li>
            <li><strong style="color: #2563eb;">Select Model:</strong> Choose from 6 AI models</li>
            <li><strong style="color: #2563eb;">View Results:</strong> Analyze detailed metrics and visualizations</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 12px; 
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05); margin-top: 2rem; 
                border-left: 4px solid #3b82f6;">
        <h3 style="color: #1e40af; margin-top: 0;">Sample Data Format</h3>
        <p style="color: #475569;">Your CSV file should have the following columns:</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    
    | Column | Description | Type |
    |--------|-------------|------|
    | age | Age in years | Integer |
    | gender | Gender (1=female, 2=male) | Binary |
    | height | Height in cm | Integer |
    | weight | Weight in kg | Float |
    | ap_hi | Systolic blood pressure | Integer |
    | ap_lo | Diastolic blood pressure | Integer |
    | cholesterol | Cholesterol (1=normal, 2=above, 3=well above) | Categorical |
    | gluc | Glucose (1=normal, 2=above, 3=well above) | Categorical |
    | smoke | Smoking status (0=no, 1=yes) | Binary |
    | alco | Alcohol intake (0=no, 1=yes) | Binary |
    | active | Physical activity (0=no, 1=yes) | Binary |
    | cardio | Disease presence (0=no, 1=yes) | Binary |
    
    **Tip:** Use the **"Download Sample Test Data"** button in the sidebar to get a ready-to-use sample dataset!
    """)
