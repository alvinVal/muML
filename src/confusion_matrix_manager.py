"""
Confusion Matrix Manager module for muML
Handles confusion matrix visualization and display
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.metrics import confusion_matrix
from typing import Dict, Any, List


class ConfusionMatrixManager:
    """Handles confusion matrix generation and visualization"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
    
    def show_confusion_matrices(self):
        """Show confusion matrices for all trained models"""
        if not hasattr(self.parent_app, 'detailed_results') or not self.parent_app.detailed_results:
            from tkinter import messagebox
            messagebox.showwarning("Missing", "Train models first to view confusion matrices.")
            return
        
        # Create popup window for confusion matrices
        confusion_window = tk.Toplevel(self.parent_app.master)
        confusion_window.title("muML - Confusion Matrices")
        confusion_window.geometry("1400x800")
        
        # Create notebook for tabs
        notebook = ttk.Notebook(confusion_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Get unique class labels from all models
        all_class_labels = set()
        for model_name, results in self.parent_app.detailed_results.items():
            if 'y_test' in results and 'y_pred' in results:
                y_test = results['y_test']
                y_pred = results['y_pred']
                all_class_labels.update(y_test)
                all_class_labels.update(y_pred)
        
        # Sort class labels for consistent display
        class_labels = sorted(list(all_class_labels))
        
        # Create a tab for each model
        for model_name, results in self.parent_app.detailed_results.items():
            if 'y_test' not in results or 'y_pred' not in results:
                continue
                
            # Create frame for this model's confusion matrix
            model_frame = ttk.Frame(notebook)
            notebook.add(model_frame, text=model_name)
            
            # Create confusion matrix
            y_test = results['y_test']
            y_pred = results['y_pred']
            
            # Calculate confusion matrix
            cm = confusion_matrix(y_test, y_pred, labels=class_labels)
            
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(10, 8), dpi=100)
            
            # Plot confusion matrix
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            ax.figure.colorbar(im, ax=ax)
            
            # Set labels
            ax.set(xticks=np.arange(cm.shape[1]),
                   yticks=np.arange(cm.shape[0]),
                   xticklabels=class_labels,
                   yticklabels=class_labels,
                   ylabel='True Label',
                   xlabel='Predicted Label')
            
            # Rotate the tick labels and set their alignment
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, format(cm[i, j], 'd'),
                           ha="center", va="center",
                           color="white" if cm[i, j] > thresh else "black")
            
            # Set title
            ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
            
            # Add accuracy information
            accuracy = results.get('accuracy', 0.0)
            f1_score = results.get('f1_weighted', 0.0)
            ax.text(0.02, 0.98, f'Accuracy: {accuracy:.3f}\nF1-Score: {f1_score:.3f}', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            
            # Create canvas and add to frame
            canvas = FigureCanvasTkAgg(fig, master=model_frame)
            canvas.draw()
            widget = canvas.get_tk_widget()
            widget.pack(fill=tk.BOTH, expand=True)
            
            # Add scrollbars if needed
            scrollbar_v = ttk.Scrollbar(model_frame, orient="vertical", command=canvas.get_tk_widget().yview)
            scrollbar_h = ttk.Scrollbar(model_frame, orient="horizontal", command=canvas.get_tk_widget().xview)
            canvas.get_tk_widget().configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
            
            # Pack scrollbars
            scrollbar_v.pack(side=tk.RIGHT, fill=tk.Y)
            scrollbar_h.pack(side=tk.BOTTOM, fill=tk.X)


