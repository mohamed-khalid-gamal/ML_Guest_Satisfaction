"""
Configuration file for Guest Satisfaction Prediction Project
"""

import os
from pathlib import Path

# Project structure
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCS_DIR = PROJECT_ROOT / "docs"

# Data files
DATASET_FILE = DATA_DIR / "GuestSatisfactionPrediction.csv"
TRAIN_DATA = DATA_DIR / "train_data.csv"
TEST_DATA = DATA_DIR / "test_data.csv"
TRAIN_DATA_REG = DATA_DIR / "train_data_reg.csv"
TEST_DATA_REG = DATA_DIR / "test_data_reg.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Text processing parameters
MAX_FEATURES_TFIDF = 20
TEXT_COLUMNS = [
    'name', 'summary', 'space', 'description', 'neighborhood_overview',
    'notes', 'transit', 'access', 'interaction', 'house_rules', 'host_about'
]

# Numerical columns for scaling
NUMERICAL_COLUMNS = [
    'nightly_price', 'price_per_stay', 'security_deposit', 'cleaning_fee',
    'extra_people', 'host_response_rate', 'latitude', 'longitude',
    'accommodates', 'bathrooms', 'bedrooms', 'beds', 'minimum_nights',
    'maximum_nights', 'availability_365', 'number_of_reviews'
]

# Categorical columns
CATEGORICAL_COLUMNS = [
    'host_is_superhost', 'host_identity_verified', 'property_type',
    'room_type', 'bed_type', 'cancellation_policy', 'instant_bookable'
]

# Target columns
TARGET_CLASSIFICATION = 'guest_satisfaction'
TARGET_REGRESSION = 'guest_satisfaction_score'

# Model configurations
MODEL_CONFIGS = {
    'random_forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'gradient_boosting': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'xgboost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
}

# Preprocessing configurations
PREPROCESSING_CONFIG = {
    'imputation_strategy': 'median',
    'scaling_method': 'standard',
    'outlier_method': 'iqr',
    'feature_selection': 'mutual_info'
}

# Streamlit app configurations
STREAMLIT_CONFIG = {
    'page_title': 'Guest Satisfaction Prediction',
    'page_icon': '🏠',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded'
}

# File paths for saved models and preprocessors
MODEL_PATHS = {
    'best_model': MODELS_DIR / 'best_model.pkl',
    'scaler': MODELS_DIR / 'scaler.pkl',
    'imputer': MODELS_DIR / 'imputer.pkl',
    'feature_selector': MODELS_DIR / 'feature_selector.pkl',
    'label_encoder': MODELS_DIR / 'label_encoder.pkl'
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': False
        }
    }
}

# Create directories if they don't exist
def create_directories():
    """Create project directories if they don't exist."""
    for directory in [DATA_DIR, MODELS_DIR, ASSETS_DIR, NOTEBOOKS_DIR, DOCS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    create_directories()
    print("Project directories created successfully!")
