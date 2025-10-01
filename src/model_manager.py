"""
Model Manager module for muML
Handles model persistence, loading, and SHAP analysis
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import pickle
import threading
import warnings
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    import shap
    has_shap = True
except ImportError:
    has_shap = False


class ModelManager:
    """Handles model persistence, loading, and SHAP analysis"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.trained_models: Dict[str, Any] = {}
        self.detailed_results: Dict[str, Dict[str, Any]] = {}
        self.all_predictions: Dict[str, pd.DataFrame] = {}
        self.predictions_df: Optional[pd.DataFrame] = None
        self.all_predictions_df: Optional[pd.DataFrame] = None
        self.custom_params: Dict[str, Dict[str, Any]] = {}
        self.search_strategies: Dict[str, tk.StringVar] = {}
        self.random_state: int = 42
    
    def save_models(self):
        """Save trained models and results to a file"""
        if not hasattr(self.parent_app, 'trained_models') or not self.parent_app.trained_models:
            messagebox.showwarning("Missing", "Train models first to save.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            title="Save Models and Results"
        )
        
        if filename:
            try:
                # Prepare data to save
                save_data = {
                    'trained_models': self.parent_app.trained_models,
                    'detailed_results': self.parent_app.detailed_results,
                    'all_predictions': self.parent_app.all_predictions,
                    'predictions_df': self.parent_app.predictions_df,
                    'all_predictions_df': self.parent_app.all_predictions_df,
                    'custom_params': self.parent_app.custom_params,
                    'search_strategies': {k: v.get() for k, v in self.parent_app.search_strategies.items()},
                    'random_state': self.parent_app.random_state,
                    'timestamp': datetime.now().isoformat(),
                    'version': '1.0'
                }
                
                # Save to file
                with open(filename, 'wb') as f:
                    pickle.dump(save_data, f)
                
                messagebox.showinfo("Success", f"Models and results saved to {filename}")
                self.parent_app._append(f"Models and results saved to {filename}\n")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save models: {e}")
                self.parent_app._append(f"Save error: {e}\n")
    
    def load_models(self):
        """Load trained models and results from a file"""
        filename = filedialog.askopenfilename(
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            title="Load Models and Results"
        )
        
        if filename:
            try:
                # Load from file
                with open(filename, 'rb') as f:
                    load_data = pickle.load(f)
                
                # Validate version compatibility
                if 'version' not in load_data:
                    messagebox.showwarning("Warning", "This file was created with an older version. Some features may not work correctly.")
                
                # Load data
                self.parent_app.trained_models = load_data.get('trained_models', {})
                self.parent_app.detailed_results = load_data.get('detailed_results', {})
                self.parent_app.all_predictions = load_data.get('all_predictions', {})
                self.parent_app.predictions_df = load_data.get('predictions_df', None)
                self.parent_app.all_predictions_df = load_data.get('all_predictions_df', None)
                self.parent_app.custom_params = load_data.get('custom_params', {})
                
                # Load search strategies
                search_strategies_data = load_data.get('search_strategies', {})
                for name, strategy in search_strategies_data.items():
                    if name in self.parent_app.search_strategies:
                        self.parent_app.search_strategies[name].set(strategy)
                
                # Load random state
                if 'random_state' in load_data:
                    self.parent_app.random_state = load_data['random_state']
                    self.parent_app.random_state_var.set(str(self.parent_app.random_state))
                
                # Update results table if we have results
                if self.parent_app.detailed_results:
                    # Create a simple results dataframe for display
                    results_data = []
                    for name, results in self.parent_app.detailed_results.items():
                        results_data.append({
                            'Classifier': name,
                            'Accuracy': results.get('accuracy', 0),
                            'F1-Score': results.get('f1_weighted', 0)
                        })
                    
                    results_df = pd.DataFrame(results_data)
                    self.parent_app.last_results_df = results_df  # Store for charts button
                    self.parent_app.results_table.show_results(results_df)
                
                # Show loaded information
                timestamp = load_data.get('timestamp', 'Unknown')
                model_count = len(self.parent_app.trained_models)
                
                messagebox.showinfo("Success", f"Loaded {model_count} models from {filename}\nSaved on: {timestamp}")
                self.parent_app._append(f"Loaded {model_count} models from {filename}\n")
                self.parent_app._append(f"Models loaded: {', '.join(self.parent_app.trained_models.keys())}\n")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load models: {e}")
                self.parent_app._append(f"Load error: {e}\n")
    
    def show_feature_importance(self):
        """Show SHAP feature importance analysis for top 4 models"""
        if not has_shap:
            messagebox.showerror("Error", "SHAP is not installed. Please install it with: pip install shap")
            return
        
        if self.parent_app.df is None:
            messagebox.showwarning("Missing", "Load a CSV file first.")
            return
        
        # Get selected features
        features = self.parent_app.feature_selector.get_selected_features()
        if not features:
            messagebox.showwarning("Missing", "Select at least one feature.")
            return
        
        target = self.parent_app.target_var.get().strip()
        if not target:
            messagebox.showwarning("Missing", "Select a target column.")
            return
        
        # Check if we have a trained model
        if not hasattr(self.parent_app, 'detailed_results') or not self.parent_app.detailed_results:
            messagebox.showwarning("Missing", "Train a model first to analyze feature importance.")
            return
        
        # Check if we have any valid models for SHAP analysis
        valid_models = ["Random Forest", "Logistic Regression", "SVM", "Decision Tree", "AdaBoost", "KNN", "Naive Bayes", "XGBoost"]
        available_models = [name for name in self.parent_app.detailed_results.keys() if name in valid_models]
        
        if not available_models:
            messagebox.showwarning("Missing", "No models available for SHAP analysis. Train some models first.")
            return
        
        # Get top 4 models by F1 score
        model_scores = []
        for name in available_models:
            if name in self.parent_app.detailed_results:
                results = self.parent_app.detailed_results[name]
                f1_score = results.get('f1_weighted', 0)
                model_scores.append((name, f1_score))
        
        # Sort by F1 score and take top 4
        model_scores.sort(key=lambda x: x[1], reverse=True)
        top_models = [name for name, score in model_scores[:4]]
        
        if not top_models:
            messagebox.showerror("Error", "No trained models found for feature importance analysis.")
            return
        
        # Show loading message
        self.parent_app._append(f"Calculating SHAP feature importance for top {len(top_models)} models...\n")
        
        # Run SHAP analysis in a separate thread
        def shap_worker():
            try:
                # Prepare data
                X = self.parent_app.df[features]
                y = self.parent_app.df[target]
                
                # Convert target to integer labels for SHAP compatibility
                from sklearn.preprocessing import LabelEncoder
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(y.astype(str))
                
                # Handle missing values
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy="mean")
                X_imputed = imputer.fit_transform(X)
                
                # Scale features
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_imputed)
                
                # Get model classes
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.linear_model import LogisticRegression
                from sklearn.svm import SVC
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.ensemble import AdaBoostClassifier
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.naive_bayes import GaussianNB
                
                # Map model names to sklearn classes
                model_classes = {
                    "Random Forest": RandomForestClassifier,
                    "Logistic Regression": LogisticRegression,
                    "SVM": SVC,
                    "Decision Tree": DecisionTreeClassifier,
                    "AdaBoost": AdaBoostClassifier,
                    "KNN": KNeighborsClassifier,
                    "Naive Bayes": GaussianNB
                }
                
                # Add XGBoost if available
                try:
                    from xgboost import XGBClassifier
                    model_classes["XGBoost"] = XGBClassifier
                except ImportError:
                    pass
                
                # Process each top model
                shap_results = []
                for model_name in top_models:
                    if model_name not in model_classes:
                        self.parent_app.master.after(0, lambda name=model_name: self.parent_app._append(f"SHAP analysis not supported for {name}\n"))
                        continue
                    
                    try:
                        # Train a simple version of the model
                        model_class = model_classes[model_name]
                        
                        # Special handling for different models
                        if model_name == "XGBoost":
                            model = model_class(random_state=self.parent_app.random_state, eval_metric="mlogloss")
                        elif model_name == "KNN":
                            # KNN doesn't support random_state
                            model = model_class()
                        else:
                            model = model_class(random_state=self.parent_app.random_state)
                        
                        model.fit(X_scaled, y_encoded)
                        
                        # Create SHAP explainer with special handling for different models
                        if model_name == "XGBoost":
                            # Use TreeExplainer for XGBoost (more efficient)
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(X_scaled[:100])
                        elif model_name == "SVM":
                            # SVM needs special handling - ensure probability=True and use KernelExplainer
                            if not hasattr(model, 'predict_proba'):
                                # Retrain with probability=True
                                from sklearn.svm import SVC
                                model = SVC(random_state=self.parent_app.random_state, probability=True)
                                model.fit(X_scaled, y_encoded)
                            explainer = shap.KernelExplainer(model.predict_proba, X_scaled[:50])
                            shap_values = explainer.shap_values(X_scaled[:50])
                        elif model_name == "KNN":
                            # KNN needs special handling - use KernelExplainer
                            explainer = shap.KernelExplainer(model.predict_proba, X_scaled[:50])
                            shap_values = explainer.shap_values(X_scaled[:50])
                        else:
                            # Use Explainer for other models
                            explainer = shap.Explainer(model, X_scaled)
                            shap_values = explainer(X_scaled[:100])
                        
                        # Check if SHAP values were generated successfully
                        if shap_values is not None:
                            shap_results.append((model_name, shap_values))
                            self.parent_app.master.after(0, lambda name=model_name: self.parent_app._append(f"Generated SHAP values for {name}\n"))
                        else:
                            self.parent_app.master.after(0, lambda name=model_name: self.parent_app._append(f"Failed to generate SHAP values for {name}\n"))
                    
                    except Exception as model_error:
                        self.parent_app.master.after(0, lambda name=model_name, err=model_error: self.parent_app._append(f"Error with {name}: {err}\n"))
                        continue
                
                # Show SHAP plots for all successful models
                if shap_results:
                    self.parent_app.master.after(0, lambda: self._display_multiple_shap_plots(shap_results, features))
                else:
                    self.parent_app.master.after(0, lambda: self.parent_app._append("No SHAP values could be generated for any model\n"))
                
            except Exception as e:
                error_msg = f"SHAP analysis error: {e}\n"
                self.parent_app.master.after(0, lambda msg=error_msg: self.parent_app._append(msg))
        
        threading.Thread(target=shap_worker, daemon=True).start()
    
    def _display_multiple_shap_plots(self, shap_results, features):
        """Display SHAP feature importance as colorful bar charts for multiple models"""
        try:
            # Create popup window for SHAP plots
            shap_window = tk.Toplevel(self.parent_app.master)
            shap_window.title("muML - Feature Importance - Top Models")
            shap_window.geometry("1400x800")
            
            # Create matplotlib figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=100)
            axes = axes.flatten()  # Flatten for easier indexing
            
            # Color palette for different features
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F']
            
            # Process each model's SHAP values
            for i, (model_name, shap_values) in enumerate(shap_results[:4]):  # Limit to 4 plots
                if i >= 4:
                    break
                
                ax = axes[i]
                
                try:
                    # Handle different SHAP value formats
                    if model_name == "XGBoost" and isinstance(shap_values, list):
                        # XGBoost returns a list of arrays for multi-class
                        if len(shap_values) > 1:
                            # Multi-class: average across classes
                            shap_values_avg = np.mean(shap_values, axis=0)
                        else:
                            # Binary classification
                            shap_values_avg = shap_values[0]
                        feature_importance = np.abs(shap_values_avg).mean(axis=0)
                    else:
                        # Standard SHAP values
                        if hasattr(shap_values, 'values'):
                            shap_values_array = shap_values.values
                        else:
                            shap_values_array = shap_values
                        
                        # Calculate feature importance
                        if len(shap_values_array.shape) > 2:
                            feature_importance = np.abs(shap_values_array).mean(axis=(0, 1))
                        else:
                            feature_importance = np.abs(shap_values_array).mean(axis=0)
                    
                    # Get top 5 features
                    num_features = min(len(features), len(feature_importance))
                    if num_features == 0:
                        ax.text(0.5, 0.5, f'No features available for {model_name}', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{model_name} - No Data')
                        continue
                    
                    # Create feature importance data
                    feature_data = []
                    for j in range(num_features):
                        try:
                            feature_data.append({
                                'Feature': features[j],
                                'Importance': feature_importance[j]
                            })
                        except IndexError:
                            break
                    
                    # Sort by importance and take top 5
                    feature_data.sort(key=lambda x: x['Importance'], reverse=True)
                    top_features = feature_data[:5]
                    
                    if not top_features:
                        ax.text(0.5, 0.5, f'No valid features for {model_name}', 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(f'{model_name} - No Data')
                        continue
                    
                    # Extract data for plotting
                    feature_names = [item['Feature'] for item in top_features]
                    importance_values = [item['Importance'] for item in top_features]
                    
                    # Create horizontal bar chart with different colors
                    bars = ax.barh(range(len(feature_names)), importance_values, 
                                  color=colors[:len(feature_names)])
                    
                    # Customize the plot
                    ax.set_yticks(range(len(feature_names)))
                    ax.set_yticklabels(feature_names, fontsize=10)
                    ax.set_xlabel('Mean |SHAP value|', fontsize=11)
                    ax.set_title(f'{model_name} - Top {len(top_features)} Features', 
                                fontsize=12, fontweight='bold')
                    ax.invert_yaxis()  # Highest importance at top
                    
                    # Add value labels on bars
                    for j, (bar, value) in enumerate(zip(bars, importance_values)):
                        ax.text(bar.get_width() + max(importance_values) * 0.01, 
                               bar.get_y() + bar.get_height()/2, 
                               f'{value:.4f}', 
                               va='center', ha='left', fontsize=9)
                    
                    # Add grid for better readability
                    ax.grid(axis='x', alpha=0.3)
                    
                except Exception as plot_error:
                    ax.text(0.5, 0.5, f'Error creating plot for {model_name}:\n{plot_error}', 
                           ha='center', va='center', transform=ax.transAxes, fontsize=10)
                    ax.set_title(f'{model_name} - Error')
            
            # Hide unused subplots
            for i in range(len(shap_results), 4):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            
            # Embed in tkinter
            canvas = FigureCanvasTkAgg(fig, master=shap_window)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Add scrollbars
            scrollbar_v = ttk.Scrollbar(shap_window, orient="vertical", command=canvas.get_tk_widget().yview)
            scrollbar_h = ttk.Scrollbar(shap_window, orient="horizontal", command=canvas.get_tk_widget().xview)
            canvas.get_tk_widget().configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
            
            scrollbar_v.pack(side=tk.RIGHT, fill=tk.Y)
            scrollbar_h.pack(side=tk.BOTTOM, fill=tk.X)
            
            self.parent_app._append(f"SHAP feature importance analysis completed for {len(shap_results)} models\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display SHAP plots: {e}")
            self.parent_app._append(f"SHAP display error: {e}\n")
