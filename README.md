# muML - Multiple Machine Learning Algorithms

A user-friendly desktop application for training and comparing multiple machine learning algorithms on your datasets. muML provides an intuitive GUI for data preprocessing, model training, and performance evaluation.

## Features

- **Easy Data Loading**: Load CSV files with automatic column detection
- **Flexible Feature Selection**: Choose target and feature columns with checkboxes
- **Multiple Algorithms**: Train 7+ algorithms including Random Forest, SVM, XGBoost, and more
- **Real-time Progress**: See which model is training with timing and accuracy updates
- **Interactive Results**: Click any result row to see detailed per-class metrics
- **Visual Charts**: Popup charts showing model performance comparisons
- **Data Preprocessing**: Automatic missing value imputation and feature scaling
- **Export Results**: Save predictions and detailed metrics to CSV

## Quick Start

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Run muML**
```bash
python -m src.gui_app
```

3. **Use the Interface**
   - Load your CSV file (defaults to `tree_features.csv` if present)
   - Select target column (defaults to `group_id`)
   - Choose features using checkboxes
   - Pick algorithms to train
   - Click "Run" and watch real-time progress
   - Double-click result rows for detailed metrics

## Supported Algorithms

- **AdaBoost**: Adaptive Boosting with decision trees
- **Decision Tree**: Single decision tree with balanced class weights
- **Random Forest**: Ensemble of decision trees
- **K-Nearest Neighbors**: Instance-based learning
- **Naive Bayes**: Gaussian Naive Bayes classifier
- **Support Vector Machine**: SVM with RBF and linear kernels
- **Logistic Regression**: Linear classifier with regularization
- **XGBoost**: Gradient boosting (optional dependency)

## Project Structure

```
muML/
├── src/
│   ├── __init__.py
│   ├── data.py           # CSV loading utilities
│   ├── preprocess.py     # Data preprocessing (imputation, scaling)
│   ├── models.py         # Algorithm registry and hyperparameter grids
│   ├── train_eval.py     # Training pipeline with progress tracking
│   └── gui_app.py        # Main Tkinter desktop application
├── requirements.txt      # Python dependencies
└── README.md
```

## Data Requirements

- **Input**: CSV files with numeric features
- **Target**: Categorical target variable for classification
- **Missing Values**: Automatically imputed with mean values
- **Scaling**: Features are standardized before training

## Advanced Features

- **Hyperparameter Tuning**: Grid search and randomized search for optimal parameters
- **Cross-Validation**: 5-fold CV for robust performance estimation
- **Class Balancing**: Automatic handling of imbalanced datasets
- **Progress Tracking**: Real-time updates during training
- **Detailed Metrics**: Per-class precision, recall, F1-score, and support

## Installation Notes

- **XGBoost**: Optional dependency. If not installed, simply deselect it in the GUI
- **Python 3.8+**: Required for type hints and modern features
- **Memory**: Recommended 4GB+ RAM for large datasets

## License

This project is open source and available under the MIT License.
