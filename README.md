# muML - Multiple Machine Learning Algorithms

A desktop application for training and comparing machine learning algorithms on tabular datasets.

## Features

- **Data Management**: Load CSV files, select features, handle missing values
- **Algorithm Support**: Random Forest, SVM, XGBoost, Decision Tree, AdaBoost, KNN, Naive Bayes, Logistic Regression
- **Hyperparameter Optimization**: Grid search and randomized search with detailed results
- **Comprehensive Metrics**: Overall accuracy, mean class accuracy, Kappa, F1-score, producer's accuracy
- **Interactive Results**: Compare models side-by-side with visual indicators
- **Analysis Tools**: Feature importance, confusion matrices, performance charts
- **Model Persistence**: Save/load trained models and configurations
- **Resource Control**: Configure CPU usage and ensure reproducibility with random state

## Quick Start

```bash
pip install -r requirements.txt
python -m src.gui_app
```

## Data Requirements

- CSV files with numeric features
- Categorical target column
- Missing values allowed (automatically imputed)

## Supported Algorithms

- **AdaBoost**: Adaptive Boosting with decision trees
- **Decision Tree**: Single decision tree with balanced class weights
- **Random Forest**: Ensemble of decision trees
- **K-Nearest Neighbors**: Instance-based learning
- **Naive Bayes**: Gaussian Naive Bayes classifier
- **Support Vector Machine**: SVM with RBF and linear kernels
- **Logistic Regression**: Linear classifier with regularization
- **XGBoost**: Gradient boosting with SHAP support (optional dependency)

## Project Structure

```
muML/
├── src/
│   ├── __init__.py
│   ├── data.py           # CSV loading utilities
│   ├── preprocess.py     # Data preprocessing (imputation, scaling)
│   ├── models.py         # Algorithm registry and hyperparameter grids
│   ├── train_eval.py     # Training pipeline with progress tracking
│   ├── charts.py         # Visualization and charting functions
│   ├── results_viewer.py # Results display and filtering
│   ├── model_manager.py  # Model persistence and loading
│   └── gui_app.py        # Main Tkinter desktop application
├── requirements.txt      # Python dependencies
└── README.md
```

## Usage

1. Load a CSV file (defaults to `species_features.csv`)
2. Select target and features
3. Choose algorithms and configure hyperparameters
4. Click "🚀 Run Training" 
5. Explore results with "📊 View Results", "📈 Show Charts", or "🔢 Confusion Matrices"
6. Save your work with "💾 Save Models" and "💾 Save Predictions"

## Installation Notes

- **Dependencies**: pandas, numpy, scikit-learn, matplotlib
- **Optional**: XGBoost, SHAP (for feature importance)
- **Requirements**: Python 3.8+, 4GB+ RAM recommended

## License

This project is open source and available under the MIT License.
