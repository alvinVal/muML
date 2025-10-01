"""
Prediction Manager module for muML
Handles prediction functionality and "Predict All" feature
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from src.results_viewer import PredictionsViewer


class PredictionManager:
    """Handles prediction functionality and model management"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
        self.trained_models: Dict[str, Any] = {}
        self.all_predictions_df: Optional[pd.DataFrame] = None
    
    def predict_all_instances(self):
        """Predict on all instances in the dataset using selected trained models"""
        if not hasattr(self.parent_app, 'trained_models') or not self.parent_app.trained_models:
            messagebox.showwarning("Missing", "Train models first to make predictions.")
            return
        
        if self.parent_app.df is None:
            messagebox.showwarning("Missing", "Load a dataset first.")
            return
        
        # Get selected features
        selected_features = self.parent_app.feature_selector.get_selected_features()
        if not selected_features:
            messagebox.showwarning("Missing", "Select at least one feature.")
            return
        
        # Get target and ID columns
        target = self.parent_app.target_var.get()
        id_col = self.parent_app.id_var.get() if self.parent_app.id_var.get() else None
        
        if not target:
            messagebox.showwarning("Missing", "Select a target column.")
            return
        
        # Get selected algorithms from main window checkboxes
        selected_models = self.parent_app.algorithm_selector.get_selected_algorithms()
        if not selected_models:
            messagebox.showwarning("Missing", "Select at least one algorithm to use for predictions.")
            return
        
        # Filter to only include models that were actually trained
        available_models = [model for model in selected_models if model in self.parent_app.trained_models]
        if not available_models:
            messagebox.showwarning("Missing", "None of the selected algorithms have been trained yet.")
            return
        
        # Show which models will be used
        self.parent_app._append(f"Using selected models for prediction: {', '.join(available_models)}\n")
        
        # Run predictions with selected models
        self._run_predictions_with_models(available_models, selected_features, target, id_col)
    
    def _run_predictions_with_models(self, selected_models, selected_features, target, id_col):
        """Run predictions with selected models using the same preprocessing as training"""
        try:
            # Create a copy of the dataframe
            df_copy = self.parent_app.df.copy()
            
            # Get actual target values for comparison
            actual_values = df_copy[target].values
            
            # Get unique classes from the actual data
            unique_classes = sorted(df_copy[target].unique())
            self.parent_app._append(f"Dataset classes: {unique_classes}\n")
            
            # Apply the same preprocessing as training manually
            # This ensures we use the same label encoder and scaler
            from sklearn.preprocessing import LabelEncoder, StandardScaler
            from sklearn.impute import SimpleImputer
            
            # Prepare features
            X = df_copy[selected_features].copy()
            
            # Handle missing values (same as training)
            imputer = SimpleImputer(strategy="mean")
            X_imputed = imputer.fit_transform(X)
            X_imputed = pd.DataFrame(X_imputed, columns=selected_features, index=X.index)
            
            # Scale features (same as training)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_imputed)
            
            # Encode target labels (same as training)
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(df_copy[target])
            
            self.parent_app._append(f"Using label encoder classes: {label_encoder.classes_}\n")
            
            # Make predictions with selected models
            predictions_data = []
            model_names = selected_models
            
            for model_name in selected_models:
                model = self.parent_app.trained_models[model_name]
                try:
                    # Make predictions (these will be encoded)
                    y_pred_encoded = model.predict(X_scaled)
                    
                    # Decode predictions back to original labels
                    y_pred = label_encoder.inverse_transform(y_pred_encoded)
                    
                    # Check for any remaining invalid predictions
                    invalid_predictions = []
                    for i, pred in enumerate(y_pred):
                        if pred not in unique_classes:
                            invalid_predictions.append((i, pred))
                    
                    if invalid_predictions:
                        self.parent_app._append(f"Warning: {model_name} made {len(invalid_predictions)} invalid predictions after decoding\n")
                        self.parent_app._append(f"Invalid predictions: {[pred for _, pred in invalid_predictions[:10]]}{'...' if len(invalid_predictions) > 10 else ''}\n")
                    
                    # Get prediction probabilities if available
                    try:
                        y_proba = model.predict_proba(X_scaled)
                        confidence = np.max(y_proba, axis=1)
                    except:
                        confidence = np.ones(len(y_pred))  # Default confidence
                    
                    # Store predictions
                    for i, (pred, conf, actual) in enumerate(zip(y_pred, confidence, actual_values)):
                        predictions_data.append({
                            'model': model_name,
                            'instance_id': df_copy.index[i] if id_col is None else df_copy.iloc[i][id_col],
                            'prediction': pred,
                            'actual': actual,
                            'confidence': conf,
                            'is_correct': pred == actual
                        })
                    
                    self.parent_app._append(f"Generated predictions for {model_name}: {len(y_pred)} instances\n")
                    
                except Exception as e:
                    self.parent_app._append(f"Error predicting with {model_name}: {e}\n")
                    continue
            
            # Create predictions dataframe
            self.all_predictions_df = pd.DataFrame(predictions_data)
            
            # Show results
            self._show_all_predictions_results()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to make predictions: {e}")
            self.parent_app._append(f"Prediction error: {e}\n")
    
    def _show_all_predictions_results(self):
        """Show results of predictions on all instances"""
        if self.all_predictions_df is None:
            return
        
        # Create results window
        predictions_window = tk.Toplevel(self.parent_app.master)
        predictions_window.title("muML - Predictions on All Instances")
        predictions_window.geometry("1600x800")
        
        # Use PredictionsViewer to display results
        PredictionsViewer(predictions_window, self.all_predictions_df)
        
        self.parent_app._append("Predictions on all instances completed successfully!\n")
    
    def save_all_predictions(self):
        """Save predictions on all instances to CSV"""
        if self.all_predictions_df is None:
            messagebox.showwarning("Missing", "No predictions to save.")
            return
        
        from tkinter import filedialog
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save All Predictions"
        )
        
        if filename:
            try:
                self.all_predictions_df.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Predictions saved to {filename}")
                self.parent_app._append(f"Predictions saved to {filename}\n")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save predictions: {e}")
                self.parent_app._append(f"Save error: {e}\n")
