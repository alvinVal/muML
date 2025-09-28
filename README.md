# muML - Multiple Machine Learning Algorithms

A user-friendly desktop application for training and comparing multiple machine learning algorithms on your datasets. muML provides an intuitive GUI for data preprocessing, model training, and performance evaluation.

## Features

- **Easy Data Loading**: Load CSV files with automatic column detection
- **Flexible Feature Selection**: Choose target and feature columns with checkboxes
- **Multiple Algorithms**: Train 7+ algorithms including Random Forest, SVM, XGBoost, and more
- **Hyperparameter Optimization**: Configure and tune hyperparameters for each algorithm
- **Real-time Progress**: See which model is training with timing and accuracy updates
- **Interactive Results**: Click any result row to see detailed per-class metrics, accuracy analysis, and top 10 hyperparameter attempts
- **Results Viewer**: Side-by-side comparison of individual sample predictions across all models with color-coded correctness indicators and filtering
- **Visual Charts**: Popup charts showing model performance comparisons
- **Data Preprocessing**: Automatic missing value imputation and feature scaling
- **Feature Importance**: SHAP-based feature importance analysis and visualization
- **Resource Control**: Configure CPU usage (n_jobs) and random state for optimal performance and reproducibility
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
   - Configure hyperparameters by clicking "Configure [Algorithm]"
   - Analyze feature importance by clicking "Feature Importance" (after training)
   - View detailed results by clicking "View Results" (after training)
   - Adjust n_jobs for CPU usage control (default: 8 cores)
   - Set random state for reproducibility (default: 42)
   - Click "Run" and watch real-time progress
   - Double-click result rows for detailed metrics and hyperparameter analysis

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
- **Top 10 Analysis**: View the best 10 hyperparameter attempts for each algorithm
- **Feature Importance Analysis**: SHAP-based feature importance with interactive visualizations
- **Individual Sample Analysis**: Side-by-side comparison of predictions for each sample across all models with visual correctness indicators
- **Color-Coded Results**: Green for correct predictions, red for incorrect, with row-level highlighting
- **Advanced Filtering**: Filter results by train/test split, group, and correctness
- **Adaptive Cross-Validation**: Automatically adjusts CV folds based on smallest class size for optimal performance
- **Class Balancing**: Automatic handling of imbalanced datasets
- **Progress Tracking**: Real-time updates during training
- **Detailed Metrics**: Per-class precision, recall, F1-score, support, producer's accuracy, mean class accuracy, and kappa accuracy
- **Resource Management**: Control CPU usage with n_jobs parameter and random state for reproducibility
- **Interactive Configuration**: Separate checkboxes for algorithm selection and configure buttons for hyperparameters

### Detailed Accuracy Metrics

muML provides comprehensive accuracy analysis for each model:

- **Overall Accuracy**: Standard accuracy (correct predictions / total predictions)
- **Producer's Accuracy**: Per-class recall (how well each class is identified)
- **Mean Class Accuracy**: Average of producer's accuracies across all classes
- **Kappa Accuracy**: Cohen's Kappa coefficient (agreement beyond chance)

### Reproducibility

muML ensures reproducible results through comprehensive random state control:

- **Global Random State**: Set a single random state for all algorithms and operations
- **Model Consistency**: All models use the same random state for consistent results
- **Cross-Validation**: Random state applied to train-test splits and CV folds
- **Hyperparameter Search**: Randomized search uses controlled randomness
- **SHAP Analysis**: Feature importance analysis uses consistent random state

**Default**: Random state 42 (ensures reproducible results out of the box)

## Installation Notes

- **XGBoost**: Optional dependency. If not installed, simply deselect it in the GUI
- **SHAP**: Required for feature importance analysis. Install with: `pip install shap`
- **Python 3.8+**: Required for type hints and modern features
- **Memory**: Recommended 4GB+ RAM for large datasets
- **CPU Usage**: Default uses 8 cores; adjust n_jobs parameter as needed

## License

This project is open source and available under the MIT License.
