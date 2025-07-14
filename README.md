# Guest Satisfaction Prediction

A machine learning project that predicts guest satisfaction for Airbnb listings using various features such as property details, host information, and textual descriptions.

## 🎯 Project Overview

This project uses machine learning techniques to predict guest satisfaction based on Airbnb listing data. The model analyzes various features including:
- Property characteristics (price, location, amenities)
- Host information (response rate, superhost status)
- Textual features (descriptions, reviews, house rules)
- Booking patterns and availability

## 📊 Dataset

The dataset contains information about Airbnb listings with features such as:
- **Numerical features**: Price, response rate, availability, etc.
- **Categorical features**: Property type, location, host verification
- **Text features**: Property descriptions, house rules, host about sections
- **Target variable**: Guest satisfaction scores

## 🚀 Features

- **Data Preprocessing**: Comprehensive data cleaning and feature engineering
- **Text Analysis**: TF-IDF vectorization and sentiment analysis
- **Model Selection**: Multiple algorithms including:
  - Random Forest
  - Gradient Boosting
  - XGBoost
  - CatBoost
  - Linear Regression variants
  - Support Vector Regression
- **Web Application**: Streamlit GUI for model predictions
- **Model Persistence**: Saved models for quick inference

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/guest-satisfaction-prediction.git
cd guest-satisfaction-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download NLTK data:
```python
import nltk
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 📁 Project Structure

```
├── data/                   # Dataset files
├── models/                 # Saved model files
├── notebooks/              # Jupyter notebooks
├── src/                    # Source code
│   ├── gui.py             # Streamlit web application
│   ├── preprocessing.py   # Data preprocessing functions
│   └── ML_Project.py      # Main ML pipeline
├── docs/                   # Documentation
├── assets/                 # Images and visualizations
├── requirements.txt        # Python dependencies
├── .gitignore             # Git ignore file
└── README.md              # This file
```

## 🔧 Usage

### Training the Model
```bash
python src/ML_Project.py
```

### Running the Web Application
```bash
streamlit run src/gui.py
```

### Data Preprocessing
```bash
python src/preprocessing.py
```

## 📈 Model Performance

The project implements multiple machine learning algorithms with comprehensive evaluation metrics:
- **R² Score**: Model accuracy measurement
- **RMSE**: Root Mean Square Error
- **MAE**: Mean Absolute Error
- **Cross-validation**: K-fold validation for robust performance assessment

## 🖥️ Web Application

The Streamlit application provides:
- Interactive prediction interface
- Feature input forms
- Real-time predictions
- Model performance visualization
- User-friendly design with professional styling

## 🔍 Key Features

### Data Processing
- **Missing Value Imputation**: KNN-based imputation
- **Outlier Detection**: Statistical methods for outlier identification
- **Feature Scaling**: StandardScaler and MinMaxScaler
- **Text Processing**: TF-IDF vectorization, sentiment analysis

### Machine Learning Models
- **Classification**: Decision Tree, Random Forest, SVM
- **Regression**: Linear, Ridge, Lasso, ElasticNet
- **Ensemble Methods**: Gradient Boosting, XGBoost, CatBoost
- **Model Selection**: Grid search and cross-validation

### Feature Engineering
- **Clustering**: K-means clustering for host categorization
- **Sentiment Analysis**: VADER sentiment analysis for text features
- **Polynomial Features**: Feature interaction terms
- **Dimensionality Reduction**: PCA for feature optimization

## 📊 Visualizations

The project includes comprehensive visualizations:
- Data distribution plots
- Correlation heatmaps
- Feature importance charts
- Model performance comparisons
- Prediction accuracy plots

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- **Team SC_3** - Machine Learning Project

## 🙏 Acknowledgments

- Dataset providers
- Open source machine learning libraries
- Streamlit for the web application framework
- NLTK for natural language processing

## 📞 Contact

For questions or suggestions, please open an issue or contact the team.

---

⭐ **Star this repository if you find it helpful!**
