"""
Utility functions for the Guest Satisfaction Prediction project.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')

def load_data(file_path):
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def get_basic_info(df):
    """Get basic information about the dataset."""
    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Data types:\n{df.dtypes}")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicated rows: {df.duplicated().sum()}")
    return df.describe()

def plot_missing_values(df):
    """Plot missing values heatmap."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.isnull(), cbar=True, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, numerical_cols):
    """Plot correlation matrix for numerical columns."""
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()

def plot_distribution(df, column):
    """Plot distribution of a column."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    df[column].hist(bins=30, edgecolor='black')
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    df.boxplot(column=column)
    plt.title(f'Box Plot of {column}')
    
    plt.tight_layout()
    plt.show()

def evaluate_classification_model(y_true, y_pred, y_prob=None):
    """Evaluate classification model performance."""
    print("Classification Report:")
    print(classification_report(y_true, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        print(f"AUC-ROC Score: {auc:.4f}")
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

def evaluate_regression_model(y_true, y_pred):
    """Evaluate regression model performance."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print("Regression Metrics:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Actual vs Predicted Values')
    
    plt.subplot(1, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    
    plt.tight_layout()
    plt.show()
    
    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

def plot_feature_importance(model, feature_names, top_n=20):
    """Plot feature importance."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top {top_n} Feature Importances')
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("Model doesn't have feature_importances_ attribute")

def save_results(results, filename):
    """Save results to a file."""
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    filepath = output_dir / filename
    with open(filepath, 'w') as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print(f"Results saved to {filepath}")

def clean_price_columns(df, price_columns):
    """Clean price columns by removing $ and converting to float."""
    for col in price_columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].replace('[\$,]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def clean_percentage_columns(df, percentage_columns):
    """Clean percentage columns by removing % and converting to float."""
    for col in percentage_columns:
        if col in df.columns and df[col].dtype == 'object':
            df[col] = df[col].str.replace('%', '').astype(float) / 100
    return df

def detect_outliers(df, column, method='iqr'):
    """Detect outliers using IQR method."""
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers
    else:
        # Z-score method
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = df[z_scores > 3]
        return outliers

def print_model_comparison(models_results):
    """Print comparison of different models."""
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    for model_name, metrics in models_results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

def create_project_structure():
    """Create the standard project structure."""
    directories = [
        'data', 'models', 'notebooks', 'src', 'docs', 'assets',
        'results', 'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    print("Project structure created successfully!")

if __name__ == "__main__":
    create_project_structure()
