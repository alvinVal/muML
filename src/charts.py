"""
Charts module for muML
Handles visualization and chart generation
"""

import tkinter as tk
from tkinter import ttk
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from typing import Optional


class ChartsManager:
    """Handles chart generation and visualization"""
    
    def __init__(self, parent_app):
        self.parent_app = parent_app
    
    def show_charts(self):
        """Show charts from the last training run"""
        if not hasattr(self.parent_app, 'last_results_df') or self.parent_app.last_results_df is None or self.parent_app.last_results_df.empty:
            from tkinter import messagebox
            messagebox.showwarning("Missing", "Train models first to view charts.")
            return
        
        self.draw_charts(self.parent_app.last_results_df)
    
    def draw_charts(self, results_df: pd.DataFrame):
        """Draw performance charts"""
        if results_df.empty:
            return
        
        # Create popup window for charts
        chart_window = tk.Toplevel(self.parent_app.master)
        chart_window.title("muML - Model Performance Charts")
        chart_window.geometry("1200x700")
        
        # Accuracy and F1 bar charts
        fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=100)
        ax0, ax1 = ax
        x = list(results_df["Classifier"])
        acc = list(results_df["Accuracy"])
        f1 = list(results_df["F1-Score"])
        ax0.bar(x, acc, color="#4C78A8")
        ax0.set_title("Accuracy", fontsize=16, fontweight='bold')
        ax0.set_ylim(0, 1)
        ax0.tick_params(axis='x', rotation=45, labelsize=12)
        ax0.tick_params(axis='y', labelsize=12)
        for i, v in enumerate(acc):
            ax0.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=11)
        ax1.bar(x, f1, color="#F58518")
        ax1.set_title("F1-Score", fontsize=16, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45, labelsize=12)
        ax1.tick_params(axis='y', labelsize=12)
        for i, v in enumerate(f1):
            ax1.text(i, v + 0.01, f"{v:.2f}", ha='center', fontsize=11)
        plt.tight_layout()
        canvas = FigureCanvasTkAgg(fig, master=chart_window)
        canvas.draw()
        widget = canvas.get_tk_widget()
        widget.pack(fill=tk.BOTH, expand=True)
