"""
Model Management Script for Guest Satisfaction Prediction

This script handles loading and saving of trained models and preprocessors.
"""

import os
import pickle
import joblib
import pandas as pd
from pathlib import Path

class ModelManager:
    """Manages loading and saving of models and preprocessors."""
    
    def __init__(self, models_dir="models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
    def save_model(self, model, filename):
        """Save a model to the models directory."""
        filepath = self.models_dir / filename
        if filename.endswith('.pkl'):
            with open(filepath, 'wb') as f:
                pickle.dump(model, f)
        elif filename.endswith('.joblib'):
            joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filename):
        """Load a model from the models directory."""
        filepath = self.models_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Model file {filepath} not found")
            
        if filename.endswith('.pkl'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif filename.endswith('.joblib'):
            return joblib.load(filepath)
        else:
            raise ValueError("Unsupported file format. Use .pkl or .joblib")
            
    def list_models(self):
        """List all available models."""
        models = []
        for file in self.models_dir.glob("*"):
            if file.suffix in ['.pkl', '.joblib']:
                models.append(file.name)
        return models
        
    def get_model_info(self, filename):
        """Get information about a model file."""
        filepath = self.models_dir / filename
        if filepath.exists():
            return {
                'filename': filename,
                'size': filepath.stat().st_size,
                'modified': filepath.stat().st_mtime
            }
        return None

# Example usage
if __name__ == "__main__":
    manager = ModelManager()
    
    # List all available models
    print("Available models:")
    for model in manager.list_models():
        print(f"  - {model}")
