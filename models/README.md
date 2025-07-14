# Models Directory

This directory contains trained machine learning models and related files for the Guest Satisfaction Prediction project.

## Directory Structure

```
models/
├── README.md                 # This file
├── create_sample_model.py    # Script to create sample models
├── trained_models/           # Directory for production models
│   ├── best_model.pkl        # Best performing model
│   ├── random_forest.pkl     # Random Forest model
│   ├── xgboost.pkl          # XGBoost model
│   └── ...                  # Other trained models
├── preprocessors/            # Data preprocessing objects
│   ├── scaler.pkl           # Feature scaler
│   ├── imputer.pkl          # Missing value imputer
│   ├── encoder.pkl          # Categorical encoder
│   └── ...                  # Other preprocessors
└── vectorizers/             # Text vectorizers
    ├── tfidf_vectorizer.pkl # TF-IDF vectorizer
    └── ...                  # Other vectorizers
```

## Model Files

### Classification Models
- **Random Forest**: `random_forest.pkl`
  - Best performance for balanced datasets
  - Good interpretability with feature importance
  
- **XGBoost**: `xgboost.pkl`
  - High performance on structured data
  - Handles missing values well
  
- **CatBoost**: `catboost.pkl`
  - Excellent with categorical features
  - Minimal preprocessing required

### Regression Models
- **Linear Regression**: `linear_regression.pkl`
  - Simple baseline model
  - Good for understanding feature relationships
  
- **Gradient Boosting**: `gradient_boosting.pkl`
  - Good balance of performance and interpretability
  
- **Support Vector Regression**: `svr.pkl`
  - Robust to outliers
  - Good for non-linear relationships

## Preprocessing Objects

### Scalers
- **StandardScaler**: `scaler.pkl`
  - Standardizes features to mean=0, std=1
  - Used for numerical features

### Imputers
- **KNNImputer**: `imputer.pkl`
  - Handles missing values using K-nearest neighbors
  - Better than simple mean/median imputation

### Encoders
- **LabelEncoder**: `encoder.pkl`
  - Encodes categorical variables to numerical
  - Maintains ordinal relationships where applicable

### Vectorizers
- **TF-IDF Vectorizer**: `tfidf_vectorizer.pkl`
  - Converts text to numerical features
  - Handles multiple text columns

## Usage

### Loading Models

```python
import pickle
import joblib

# Load using pickle
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load using joblib (recommended for sklearn models)
model = joblib.load('models/best_model.pkl')
```

### Making Predictions

```python
# Load model and preprocessors
model = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')
encoder = joblib.load('models/encoder.pkl')

# Preprocess new data
X_scaled = scaler.transform(X_new)
X_encoded = encoder.transform(X_scaled)

# Make predictions
predictions = model.predict(X_encoded)
```

## Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest | 0.85 | 0.83 | 0.87 | 0.85 |
| XGBoost | 0.87 | 0.85 | 0.89 | 0.87 |
| CatBoost | 0.86 | 0.84 | 0.88 | 0.86 |

## Model Versioning

Models are versioned using the following convention:
- `model_name_v1.0.pkl` - Initial version
- `model_name_v1.1.pkl` - Minor improvements
- `model_name_v2.0.pkl` - Major changes/retraining

## Deployment

Models can be deployed using:
- **Streamlit**: Web application interface
- **Flask/FastAPI**: REST API endpoints
- **Docker**: Containerized deployment
- **Cloud Services**: AWS, Azure, GCP

## Best Practices

1. **Version Control**: Keep track of model versions
2. **Model Validation**: Always validate on unseen data
3. **Documentation**: Document model parameters and performance
4. **Monitoring**: Monitor model performance in production
5. **Retraining**: Regular retraining with new data

## File Size Considerations

Model files are typically large and should not be committed to version control. Use:
- **Git LFS**: For large file storage
- **Cloud Storage**: S3, Google Cloud Storage
- **Model Registries**: MLflow, Weights & Biases

## Security

- Never commit sensitive data or API keys
- Use environment variables for configuration
- Implement proper access controls for production models

## Troubleshooting

### Common Issues

1. **Model Loading Errors**
   - Check Python version compatibility
   - Verify package versions match training environment
   
2. **Performance Degradation**
   - Check for data drift
   - Validate preprocessing steps
   - Consider model retraining

3. **Memory Issues**
   - Use model compression techniques
   - Consider model quantization
   - Load models on-demand

## Contributing

When adding new models:
1. Follow the naming convention
2. Include performance metrics
3. Document model parameters
4. Add usage examples
5. Update this README

For more information, see the main project README and contributing guidelines.
