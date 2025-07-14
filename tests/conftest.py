"""
Test configuration and fixtures for the test suite.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    
    data = {
        'nightly_price': np.random.normal(100, 30, 100),
        'host_response_rate': np.random.uniform(0.5, 1.0, 100),
        'accommodates': np.random.randint(1, 10, 100),
        'bathrooms': np.random.randint(1, 5, 100),
        'bedrooms': np.random.randint(1, 5, 100),
        'host_is_superhost': np.random.choice(['t', 'f'], 100),
        'property_type': np.random.choice(['Apartment', 'House', 'Condo'], 100),
        'description': ['Nice place to stay'] * 100,
        'guest_satisfaction': np.random.choice([0, 1], 100)
    }
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_config():
    """Create sample configuration for testing."""
    return {
        'numerical_columns': ['nightly_price', 'host_response_rate', 'accommodates', 'bathrooms', 'bedrooms'],
        'categorical_columns': ['host_is_superhost', 'property_type'],
        'text_columns': ['description'],
        'price_columns': ['nightly_price'],
        'percentage_columns': ['host_response_rate'],
        'imputation_strategy': 'median',
        'scaler_type': 'standard',
        'encoding_type': 'label',
        'outlier_method': 'iqr',
        'outlier_action': 'clip',
        'max_features_tfidf': 10
    }
