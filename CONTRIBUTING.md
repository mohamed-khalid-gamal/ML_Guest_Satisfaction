# Contributing to Guest Satisfaction Prediction

We welcome contributions to the Guest Satisfaction Prediction project! This document provides guidelines for contributing to the project.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Code Style](#code-style)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your feature or bug fix
4. Make your changes
5. Test your changes
6. Submit a pull request

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/guest-satisfaction-prediction.git
   cd guest-satisfaction-prediction
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

## Making Changes

1. **Create a new branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Follow the existing code structure
   - Add appropriate comments and documentation
   - Update README.md if necessary

3. **Test your changes:**
   ```bash
   python -m pytest tests/
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Use type hints where appropriate
- Keep functions small and focused

### Example Code Style

```python
def calculate_prediction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the accuracy of predictions.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
    
    Returns:
        Accuracy score as a float
    """
    return accuracy_score(y_true, y_pred)
```

## Testing

- Write unit tests for new functions
- Ensure all existing tests pass
- Test edge cases and error conditions
- Use pytest for testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run tests with coverage
python -m pytest --cov=src

# Run specific test file
python -m pytest tests/test_preprocessing.py
```

## Submitting Changes

1. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

2. **Push to your fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a pull request:**
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Select your branch
   - Fill out the pull request template

### Pull Request Guidelines

- Provide a clear title and description
- Reference any related issues
- Include screenshots if applicable
- Ensure all tests pass
- Update documentation if needed

## Reporting Issues

When reporting issues, please include:

1. **Description:** Clear description of the problem
2. **Environment:** Python version, OS, package versions
3. **Steps to reproduce:** Minimal example to reproduce the issue
4. **Expected behavior:** What should happen
5. **Actual behavior:** What actually happens
6. **Error messages:** Full error traceback if applicable

### Issue Template

```
**Description**
A clear and concise description of the issue.

**Environment**
- Python version: 3.x
- OS: [e.g., Windows 10, Ubuntu 20.04]
- Package version: x.x.x

**Steps to Reproduce**
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Error Messages**
```
[Paste any error messages here]
```

**Additional Context**
Add any other context about the problem here.
```

## Development Guidelines

### Adding New Features

1. **Model Improvements:**
   - Add new algorithms in the appropriate module
   - Update configuration files
   - Add proper documentation

2. **Data Processing:**
   - Extend the DataPreprocessor class
   - Add unit tests for new methods
   - Update the preprocessing pipeline

3. **Web Application:**
   - Follow Streamlit best practices
   - Add new pages to the navigation
   - Maintain consistent styling

### Code Review Process

1. All submissions require review
2. Maintainers will provide feedback
3. Address feedback promptly
4. Maintain a clean commit history

## Resources

- [Python Style Guide (PEP 8)](https://www.python.org/dev/peps/pep-0008/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## Questions?

If you have questions about contributing, please:

1. Check the existing documentation
2. Search for similar issues
3. Ask in the project discussions
4. Contact the maintainers

Thank you for contributing to Guest Satisfaction Prediction! 🎉
