"""
Detailed Metrics module for muML
Handles detailed metrics display and analysis
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, Any


class DetailedMetrics:
    """Handles detailed metrics display"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
    
    def show_detailed_metrics(self, classifier_name: str):
        """Show detailed metrics for a specific classifier"""
        if classifier_name not in self.parent_app.detailed_results:
            return
        
        details = self.parent_app.detailed_results[classifier_name]
        crep = details["classification_report"]
        
        # Create popup window for detailed metrics
        detail_window = tk.Toplevel(self.parent_app.master)
        detail_window.title(f"muML - Detailed Metrics - {classifier_name}")
        detail_window.geometry("1200x800")
        
        # Create scrollable text area
        text_frame = ttk.Frame(detail_window)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        text_widget = tk.Text(text_frame, wrap=tk.NONE, font=("Consolas", 9))
        scrollbar_v = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
        scrollbar_h = ttk.Scrollbar(text_frame, orient="horizontal", command=text_widget.xview)
        text_widget.configure(yscrollcommand=scrollbar_v.set, xscrollcommand=scrollbar_h.set)
        
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar_v.pack(side=tk.RIGHT, fill=tk.Y)
        scrollbar_h.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Format detailed metrics
        content = f"Detailed Metrics for {classifier_name}\n"
        content += "=" * 60 + "\n\n"
        
        # Basic metrics
        content += "Basic Performance Metrics:\n"
        content += "-" * 30 + "\n"
        content += f"Overall Accuracy: {details['accuracy']:.4f}\n"
        content += f"F1-Score (Weighted): {details['f1_weighted']:.4f}\n"
        content += f"Training Time: {details['training_time']:.2f} seconds\n\n"
        
        # Detailed accuracy metrics
        if 'detailed_accuracy_metrics' in details:
            acc_metrics = details['detailed_accuracy_metrics']
            content += "Detailed Accuracy Analysis:\n"
            content += "-" * 30 + "\n"
            content += f"Overall Accuracy: {acc_metrics['overall_accuracy']:.4f}\n"
            content += f"Mean Class Accuracy: {acc_metrics['mean_class_accuracy']:.4f}\n"
            content += f"Kappa Accuracy: {acc_metrics['kappa_accuracy']:.4f}\n\n"
            
            # Producer's accuracy for each class
            content += "Producer's Accuracy (Per-Class Recall):\n"
            content += "-" * 40 + "\n"
            class_labels = acc_metrics['class_labels']
            producer_accs = acc_metrics['producer_accuracy']
            
            for i, (class_label, producer_acc) in enumerate(zip(class_labels, producer_accs)):
                content += f"Class {class_label}: {producer_acc:.4f}\n"
            content += "\n"
        
        # Add hyperparameter optimization results
        if 'best_params' in details and 'cv_results' in details:
            content += "Hyperparameter Optimization Results:\n"
            content += "-" * 40 + "\n"
            content += f"Best Parameters: {details['best_params']}\n"
            content += f"Best CV Score: {details['cv_results']['mean_test_score'][details['cv_results']['rank_test_score'] == 1][0]:.4f}\n\n"
            
            # Show top 10 attempts
            content += "Top 10 Hyperparameter Attempts:\n"
            content += "-" * 40 + "\n"
            
            # Get CV results and sort by score
            cv_results = details['cv_results']
            mean_scores = cv_results['mean_test_score']
            params = cv_results['params']
            
            # Create list of (score, params) tuples and sort by score
            attempts = list(zip(mean_scores, params))
            attempts.sort(key=lambda x: x[0], reverse=True)
            
            content += f"{'Rank':<4} {'Score':<8} {'Parameters'}\n"
            content += "-" * 80 + "\n"
            
            for i, (score, param_set) in enumerate(attempts[:10], 1):
                # Format parameters more readably
                param_items = []
                for key, value in param_set.items():
                    param_items.append(f"{key}={value}")
                param_str = ", ".join(param_items)
                
                # If still too long, break into multiple lines
                if len(param_str) > 70:
                    content += f"{i:<4} {score:<8.4f} {param_str}\n"
                else:
                    content += f"{i:<4} {score:<8.4f} {param_str}\n"
            
            content += "\n"
        
        content += "Per-Class Metrics:\n"
        content += "-" * 40 + "\n"
        content += f"{'Class':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}\n"
        content += "-" * 40 + "\n"
        
        # Show per-class metrics
        for class_name in sorted(crep.keys()):
            if class_name in ['accuracy', 'macro avg', 'weighted avg']:
                continue
            metrics = crep[class_name]
            content += f"{class_name:<8} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1-score']:<10.4f} {metrics['support']:<8.0f}\n"
        
        # Add summary metrics
        content += "\nSummary:\n"
        content += "-" * 40 + "\n"
        if 'macro avg' in crep:
            content += f"Macro Avg:     {crep['macro avg']['precision']:.4f} {crep['macro avg']['recall']:.4f} {crep['macro avg']['f1-score']:.4f} {crep['macro avg']['support']:.0f}\n"
        if 'weighted avg' in crep:
            content += f"Weighted Avg:  {crep['weighted avg']['precision']:.4f} {crep['weighted avg']['recall']:.4f} {crep['weighted avg']['f1-score']:.4f} {crep['weighted avg']['support']:.0f}\n"
        
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)
