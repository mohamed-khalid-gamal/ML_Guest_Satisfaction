"""
Tests for data preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
from src.data_preprocessing import DataPreprocessor

class TestDataPreprocessor:
    """Test cases for DataPreprocessor class."""
    
    def test_init(self):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor()
        assert preprocessor.scalers == {}
        assert preprocessor.imputers == {}
        assert preprocessor.encoders == {}
        assert preprocessor.feature_selectors == {}
        assert preprocessor.tfidf_vectorizers == {}
    
    def test_clean_price_columns(self, sample_data):
        """Test price column cleaning."""
        preprocessor = DataPreprocessor()
        
        # Add price column with $ symbol
        sample_data['test_price'] = ['$100.00', '$200.50', '$300.75'] * 34
        
        cleaned_data = preprocessor.clean_price_columns(sample_data, ['test_price'])
        
        assert cleaned_data['test_price'].dtype in ['float64', 'int64']
        assert cleaned_data['test_price'].iloc[0] == 100.0
    
    def test_clean_percentage_columns(self, sample_data):
        """Test percentage column cleaning."""
        preprocessor = DataPreprocessor()
        
        # Add percentage column
        sample_data['test_percentage'] = ['50%', '75%', '100%'] * 34
        
        cleaned_data = preprocessor.clean_percentage_columns(sample_data, ['test_percentage'])
        
        assert cleaned_data['test_percentage'].dtype in ['float64', 'int64']
        assert cleaned_data['test_percentage'].iloc[0] == 0.5
    
    def test_handle_missing_values(self, sample_data):
        """Test missing value handling."""
        preprocessor = DataPreprocessor()
        
        # Introduce missing values
        sample_data.loc[0:5, 'nightly_price'] = np.nan
        sample_data.loc[0:3, 'host_is_superhost'] = np.nan
        
        processed_data = preprocessor.handle_missing_values(
            sample_data,
            ['nightly_price'],
            ['host_is_superhost']
        )
        
        assert processed_data['nightly_price'].isna().sum() == 0
        assert processed_data['host_is_superhost'].isna().sum() == 0
    
    def test_encode_categorical_features(self, sample_data):
        """Test categorical feature encoding."""
        preprocessor = DataPreprocessor()
        
        encoded_data = preprocessor.encode_categorical_features(
            sample_data,
            ['host_is_superhost', 'property_type']
        )
        
        assert encoded_data['host_is_superhost'].dtype in ['int64', 'int32']
        assert encoded_data['property_type'].dtype in ['int64', 'int32']
    
    def test_scale_features(self, sample_data):
        """Test feature scaling."""
        preprocessor = DataPreprocessor()
        
        scaled_data = preprocessor.scale_features(
            sample_data,
            ['nightly_price', 'accommodates']
        )
        
        # Check if values are scaled (mean should be close to 0, std close to 1)
        assert abs(scaled_data['nightly_price'].mean()) < 1e-10
        assert abs(scaled_data['nightly_price'].std() - 1.0) < 1e-10
    
    def test_sentiment_analysis(self, sample_data):
        """Test sentiment analysis application."""
        preprocessor = DataPreprocessor()
        
        # Add varied text for sentiment analysis
        sample_data['description'] = [
            'This is a great place to stay!',
            'Terrible experience, would not recommend.',
            'Average place, nothing special.'
        ] * 34
        
        sentiment_data = preprocessor.apply_sentiment_analysis(sample_data, ['description'])
        
        assert 'description_sentiment' in sentiment_data.columns
        assert sentiment_data['description_sentiment'].dtype in ['float64', 'int64']
    
    def test_tfidf_vectorization(self, sample_data):
        """Test TF-IDF vectorization."""
        preprocessor = DataPreprocessor()
        
        # Add varied text for TF-IDF
        sample_data['description'] = [
            'Beautiful apartment with great view',
            'Nice house near the beach',
            'Cozy condo in downtown'
        ] * 34
        
        tfidf_data = preprocessor.apply_tfidf_vectorization(sample_data, ['description'], max_features=5)
        
        # Check if TF-IDF columns are added
        tfidf_columns = [col for col in tfidf_data.columns if 'tfidf' in col]
        assert len(tfidf_columns) > 0
    
    def test_outlier_detection(self, sample_data):
        """Test outlier detection and handling."""
        preprocessor = DataPreprocessor()
        
        # Add outliers
        sample_data.loc[0, 'nightly_price'] = 10000  # Extreme outlier
        
        original_max = sample_data['nightly_price'].max()
        
        processed_data = preprocessor.detect_and_handle_outliers(
            sample_data,
            ['nightly_price'],
            method='iqr',
            action='clip'
        )
        
        # Check if outlier was handled
        assert processed_data['nightly_price'].max() < original_max
    
    def test_create_host_clusters(self, sample_data):
        """Test host clustering."""
        preprocessor = DataPreprocessor()
        
        clustered_data = preprocessor.create_host_clusters(
            sample_data,
            ['host_response_rate', 'accommodates'],
            n_clusters=3
        )
        
        assert 'host_cluster' in clustered_data.columns
        assert clustered_data['host_cluster'].nunique() <= 3
    
    def test_preprocessing_pipeline(self, sample_data, sample_config):
        """Test complete preprocessing pipeline."""
        preprocessor = DataPreprocessor()
        
        # Add missing values to test the pipeline
        sample_data.loc[0:5, 'nightly_price'] = np.nan
        sample_data.loc[0:3, 'host_is_superhost'] = np.nan
        
        processed_data = preprocessor.preprocess_pipeline(sample_data, sample_config)
        
        # Check if data was processed
        assert processed_data.shape[0] > 0
        assert processed_data.isna().sum().sum() == 0  # No missing values
        
        # Check if new columns were created
        assert any('tfidf' in col for col in processed_data.columns)
        assert any('sentiment' in col for col in processed_data.columns)

if __name__ == "__main__":
    pytest.main([__file__])
