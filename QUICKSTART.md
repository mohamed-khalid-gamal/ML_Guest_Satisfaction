# Quick Start Guide

Get up and running with the Guest Satisfaction Prediction project in minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/guest-satisfaction-prediction.git
cd guest-satisfaction-prediction
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download Required Data

```bash
# Download NLTK data
python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## Quick Demo

### Run the Web Application

```bash
streamlit run src/gui.py
```

This will open a web browser with the interactive prediction interface.

### Train a Model

```bash
python src/ML_Project.py
```

### Run Preprocessing

```bash
python src/preprocessing.py
```

## Project Structure

```
├── data/           # Dataset files
├── models/         # Trained models
├── notebooks/      # Jupyter notebooks
├── src/           # Source code
├── tests/         # Test files
├── docs/          # Documentation
└── assets/        # Images and visualizations
```

## Key Features

- **Machine Learning Models**: Multiple algorithms for classification and regression
- **Web Interface**: User-friendly Streamlit application
- **Data Processing**: Comprehensive preprocessing pipeline
- **Model Evaluation**: Detailed performance metrics and visualizations

## Common Commands

```bash
# Run tests
python -m pytest tests/

# Install in development mode
pip install -e .

# Create sample models
python models/create_sample_model.py

# Run specific notebook
jupyter notebook notebooks/ML_Project.ipynb
```

## Next Steps

1. **Explore the Data**: Check out the Jupyter notebooks in `notebooks/`
2. **Customize Models**: Modify parameters in `src/config.py`
3. **Add Features**: Extend the preprocessing pipeline
4. **Deploy**: Use the Streamlit app for production

## Getting Help

- Read the full [README.md](README.md)
- Check the [documentation](docs/)
- Review [contributing guidelines](CONTRIBUTING.md)
- Open an issue for bugs or questions

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure virtual environment is activated
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **NLTK Data**: Download required NLTK data as shown above
4. **Memory Issues**: Use smaller datasets for testing

### Performance Tips

- Use GPU if available for large datasets
- Enable multiprocessing for feature engineering
- Consider data sampling for initial experiments

That's it! You're ready to start predicting guest satisfaction! 🎉
