"""
Data preprocessing module for Guest Satisfaction Prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import re
import string
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Comprehensive data preprocessing pipeline."""
    
    def __init__(self):
        self.scalers = {}
        self.imputers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.tfidf_vectorizers = {}
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def clean_price_columns(self, df, price_columns):
        """Clean price columns by removing $ and converting to float."""
        for col in price_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].replace('[\$,]', '', regex=True)
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    def clean_percentage_columns(self, df, percentage_columns):
        """Clean percentage columns by removing % and converting to float."""
        for col in percentage_columns:
            if col in df.columns and df[col].dtype == 'object':
                df[col] = df[col].str.replace('%', '').astype(float) / 100
        return df
    
    def handle_missing_values(self, df, numerical_cols, categorical_cols, strategy='median'):
        """Handle missing values using different strategies."""
        # Numerical columns
        if numerical_cols:
            if strategy == 'knn':
                imputer = KNNImputer(n_neighbors=5)
            else:
                imputer = SimpleImputer(strategy=strategy)
            
            df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
            self.imputers['numerical'] = imputer
        
        # Categorical columns
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
            self.imputers['categorical'] = cat_imputer
        
        return df
    
    def encode_categorical_features(self, df, categorical_cols, encoding_type='label'):
        """Encode categorical features."""
        if encoding_type == 'label':
            for col in categorical_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.encoders[col] = le
        elif encoding_type == 'onehot':
            # One-hot encoding
            encoded_cols = []
            for col in categorical_cols:
                if col in df.columns:
                    dummies = pd.get_dummies(df[col], prefix=col)
                    encoded_cols.extend(dummies.columns.tolist())
                    df = pd.concat([df, dummies], axis=1)
                    df.drop(col, axis=1, inplace=True)
        
        return df
    
    def scale_features(self, df, numerical_cols, scaler_type='standard'):
        """Scale numerical features."""
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        self.scalers['numerical'] = scaler
        
        return df
    
    def detect_and_handle_outliers(self, df, numerical_cols, method='iqr', action='clip'):
        """Detect and handle outliers."""
        for col in numerical_cols:
            if col in df.columns:
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    if action == 'clip':
                        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                    elif action == 'remove':
                        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                
                elif method == 'zscore':
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    if action == 'clip':
                        df.loc[z_scores > 3, col] = df[col].median()
                    elif action == 'remove':
                        df = df[z_scores <= 3]
        
        return df
    
    def apply_sentiment_analysis(self, df, text_columns):
        """Apply sentiment analysis to text columns."""
        for col in text_columns:
            if col in df.columns:
                # Clean text
                df[col] = df[col].fillna('')
                df[col] = df[col].astype(str)
                
                # Apply sentiment analysis
                sentiment_scores = df[col].apply(self.get_sentiment_score)
                df[f'{col}_sentiment'] = sentiment_scores
        
        return df
    
    def get_sentiment_score(self, text):
        """Get sentiment score for text."""
        if not text or text.isspace():
            return 0
        
        # Clean text
        text = self.clean_text(text)
        
        # Get sentiment
        sentiment = self.sentiment_analyzer.polarity_scores(text)
        return sentiment['compound']
    
    def clean_text(self, text):
        """Clean text data."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def apply_tfidf_vectorization(self, df, text_columns, max_features=20):
        """Apply TF-IDF vectorization to text columns."""
        tfidf_features = []
        
        for col in text_columns:
            if col in df.columns:
                # Clean and prepare text
                df[col] = df[col].fillna('')
                df[col] = df[col].astype(str)
                
                # Apply TF-IDF
                tfidf = TfidfVectorizer(
                    max_features=max_features,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                
                tfidf_matrix = tfidf.fit_transform(df[col])
                
                # Create feature names
                feature_names = [f'{col}_tfidf_{i}' for i in range(tfidf_matrix.shape[1])]
                
                # Create DataFrame
                tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)
                tfidf_features.append(tfidf_df)
                
                # Save vectorizer
                self.tfidf_vectorizers[col] = tfidf
        
        # Concatenate all TF-IDF features
        if tfidf_features:
            tfidf_combined = pd.concat(tfidf_features, axis=1)
            df = pd.concat([df, tfidf_combined], axis=1)
        
        return df
    
    def create_host_clusters(self, df, host_features, n_clusters=5):
        """Create host clusters based on host features."""
        host_data = df[host_features].fillna(0)
        
        # Apply clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['host_cluster'] = kmeans.fit_predict(host_data)
        
        return df
    
    def feature_selection(self, X, y, task_type='classification', k=50):
        """Select best features using statistical tests."""
        if task_type == 'classification':
            selector = SelectKBest(score_func=f_classif, k=k)
        else:
            selector = SelectKBest(score_func=f_regression, k=k)
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        self.feature_selectors[task_type] = selector
        
        return X_selected, selected_features
    
    def create_interaction_features(self, df, feature_pairs):
        """Create interaction features."""
        for feature1, feature2 in feature_pairs:
            if feature1 in df.columns and feature2 in df.columns:
                df[f'{feature1}_{feature2}_interaction'] = df[feature1] * df[feature2]
        
        return df
    
    def preprocess_pipeline(self, df, config):
        """Complete preprocessing pipeline."""
        print("Starting preprocessing pipeline...")
        
        # Clean price columns
        if 'price_columns' in config:
            df = self.clean_price_columns(df, config['price_columns'])
            print("✓ Price columns cleaned")
        
        # Clean percentage columns
        if 'percentage_columns' in config:
            df = self.clean_percentage_columns(df, config['percentage_columns'])
            print("✓ Percentage columns cleaned")
        
        # Handle missing values
        df = self.handle_missing_values(
            df, 
            config.get('numerical_columns', []),
            config.get('categorical_columns', []),
            config.get('imputation_strategy', 'median')
        )
        print("✓ Missing values handled")
        
        # Apply sentiment analysis
        if 'text_columns' in config:
            df = self.apply_sentiment_analysis(df, config['text_columns'])
            print("✓ Sentiment analysis applied")
        
        # Apply TF-IDF vectorization
        if 'text_columns' in config:
            df = self.apply_tfidf_vectorization(
                df, 
                config['text_columns'],
                config.get('max_features_tfidf', 20)
            )
            print("✓ TF-IDF vectorization applied")
        
        # Handle outliers
        df = self.detect_and_handle_outliers(
            df,
            config.get('numerical_columns', []),
            config.get('outlier_method', 'iqr'),
            config.get('outlier_action', 'clip')
        )
        print("✓ Outliers handled")
        
        # Encode categorical features
        df = self.encode_categorical_features(
            df,
            config.get('categorical_columns', []),
            config.get('encoding_type', 'label')
        )
        print("✓ Categorical features encoded")
        
        # Scale features
        df = self.scale_features(
            df,
            config.get('numerical_columns', []),
            config.get('scaler_type', 'standard')
        )
        print("✓ Features scaled")
        
        print("Preprocessing pipeline completed!")
        return df
    
    def save_preprocessors(self, filepath):
        """Save all preprocessors to files."""
        import pickle
        
        preprocessors = {
            'scalers': self.scalers,
            'imputers': self.imputers,
            'encoders': self.encoders,
            'feature_selectors': self.feature_selectors,
            'tfidf_vectorizers': self.tfidf_vectorizers
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(preprocessors, f)
        
        print(f"Preprocessors saved to {filepath}")
    
    def load_preprocessors(self, filepath):
        """Load preprocessors from files."""
        import pickle
        
        with open(filepath, 'rb') as f:
            preprocessors = pickle.load(f)
        
        self.scalers = preprocessors.get('scalers', {})
        self.imputers = preprocessors.get('imputers', {})
        self.encoders = preprocessors.get('encoders', {})
        self.feature_selectors = preprocessors.get('feature_selectors', {})
        self.tfidf_vectorizers = preprocessors.get('tfidf_vectorizers', {})
        
        print(f"Preprocessors loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Download required NLTK data
    nltk.download('vader_lexicon', quiet=True)
    
    preprocessor = DataPreprocessor()
    print("Data preprocessor initialized successfully!")
