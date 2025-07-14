"""
Sample model creation for demonstration purposes.
This file should be run after data preprocessing to create sample models.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import joblib
from pathlib import Path

# Create sample data for demonstration
np.random.seed(42)

def create_sample_model():
    """Create a sample model for demonstration."""
    
    # Create sample data
    n_samples = 1000
    n_features = 10
    
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], n_samples)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Sample Model Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    with open(models_dir / "sample_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    print(f"\nSample model saved to {models_dir / 'sample_model.pkl'}")
    
    return model

if __name__ == "__main__":
    create_sample_model()
