"""
Refactored GUI Application for muML
Main application class using modular components
"""

from __future__ import annotations

import threading
import warnings
import pandas as pd
import numpy as np
from typing import Dict, List, Any

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from src.data import read_csv, list_columns
from src.models import get_model_registry
from src.preprocess import train_test_prepare
from src.train_eval import train_selected, build_predictions_df, build_all_predictions_df

# Import modular components
from src.ui_components import UIStyles, FeatureSelector, AlgorithmSelector, HyperparameterConfig, ResultsTable
from src.results_viewer import ResultsViewer
from src.prediction_manager import PredictionManager
from src.model_manager import ModelManager
from src.charts import ChartsManager
from src.detailed_metrics import DetailedMetrics

# Suppress warnings
warnings.simplefilter("ignore", category=UserWarning)


class MLGuiApp(ttk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.master = master
        self.pack(fill=tk.BOTH, expand=True)

        # Core data attributes
        self.df: pd.DataFrame | None = None
        self.columns: List[str] = []
        self.random_state: int = 42
        self.algos = get_model_registry(self.random_state)
        
        # Results and predictions
        self.predictions_df: pd.DataFrame | None = None
        self.detailed_results: Dict[str, Dict[str, Any]] = {}
        self.all_predictions: Dict[str, pd.DataFrame] = {}
        self.trained_models: Dict[str, Any] = {}
        self.all_predictions_df: pd.DataFrame | None = None
        self.last_results_df: pd.DataFrame | None = None
        
        # Additional attributes needed for model management
        self.custom_params: Dict[str, Dict[str, Any]] = {}
        self.search_strategies: Dict[str, tk.StringVar] = {}

        # Initialize modular components
        self.ui_styles = UIStyles()
        self.prediction_manager = PredictionManager(self)
        self.model_manager = ModelManager(self)
        self.charts_manager = ChartsManager(self)
        self.detailed_metrics = DetailedMetrics(self)

        # UI Variables
        self.target_var = tk.StringVar(value="group_id")
        self.id_var = tk.StringVar(value="tree_id")
        self.test_size_var = tk.StringVar(value="0.3")
        self.stratify_var = tk.BooleanVar(value=True)
        self.n_jobs_var = tk.StringVar(value="8")
        self.random_state_var = tk.StringVar(value="42")

        self._build_ui()

    def _build_ui(self):
        """Build the main UI using modular components"""
        self.master.title("muML - Multiple Machine Learning Algorithms")
        self.master.geometry("1600x900")
        
        # Configure modern styling
        self.ui_styles.configure_styles(ttk.Style())
        
        # Set main window background
        self.master.configure(bg=self.ui_styles.colors['light'])

        # Top controls
        top = ttk.Frame(self)
        top.pack(fill=tk.X, padx=10, pady=8)

        self.file_var = tk.StringVar(value="species_features.csv")
        file_entry = ttk.Entry(top, textvariable=self.file_var, width=80, style='Modern.TEntry')
        file_entry.pack(side=tk.LEFT, padx=(0, 8))

        browse_btn = ttk.Button(top, text="📁 Browse CSV", command=self._browse_csv, style='Info.TButton')
        browse_btn.pack(side=tk.LEFT)

        # Options frame
        opts = ttk.LabelFrame(self, text="Options", style='Modern.TLabelframe')
        opts.pack(fill=tk.X, padx=10, pady=8)

        # Configure trace callbacks
        self.target_var.trace_add("write", lambda *args: self._on_target_changed())
        self.id_var.trace_add("write", lambda *args: self._on_id_changed())
        self.random_state_var.trace_add("write", lambda *args: self._on_random_state_changed())

        row1 = ttk.Frame(opts)
        row1.pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(row1, text="Target:").pack(side=tk.LEFT)
        self.target_cb = ttk.Combobox(row1, textvariable=self.target_var, state="readonly", width=30, values=self.columns, style='Modern.TCombobox')
        self.target_cb.pack(side=tk.LEFT, padx=6)

        ttk.Label(row1, text="ID column:").pack(side=tk.LEFT)
        self.id_cb = ttk.Combobox(row1, textvariable=self.id_var, state="readonly", width=20, values=self.columns, style='Modern.TCombobox')
        self.id_cb.pack(side=tk.LEFT, padx=6)

        ttk.Label(row1, text="Test size:").pack(side=tk.LEFT)
        test_entry = ttk.Entry(row1, textvariable=self.test_size_var, width=8, style='Modern.TEntry')
        test_entry.pack(side=tk.LEFT, padx=6)

        strat_cb = ttk.Checkbutton(row1, text="Stratify", variable=self.stratify_var)
        strat_cb.pack(side=tk.LEFT, padx=6)

        ttk.Label(row1, text="n_jobs:").pack(side=tk.LEFT, padx=(10, 0))
        n_jobs_entry = ttk.Entry(row1, textvariable=self.n_jobs_var, width=8, style='Modern.TEntry')
        n_jobs_entry.pack(side=tk.LEFT, padx=6)
        ttk.Label(row1, text="(uses 8 cores by default, -1 to use all cores)", font=("Arial", 8), 
                 foreground="gray").pack(side=tk.LEFT, padx=6)
        
        # Random state control
        ttk.Label(row1, text="Random State:").pack(side=tk.LEFT, padx=(20, 0))
        random_state_entry = ttk.Entry(row1, textvariable=self.random_state_var, width=8, style='Modern.TEntry')
        random_state_entry.pack(side=tk.LEFT, padx=6)
        ttk.Label(row1, text="(controls reproducibility)", font=("Arial", 8), 
                 foreground="gray").pack(side=tk.LEFT, padx=6)

        # Middle: features, algorithms, and hyperparameters
        mid = ttk.Frame(self)
        mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Features frame
        feat_frame = ttk.LabelFrame(mid, text="🎯 Features", style='Modern.TLabelframe')
        feat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 8))
        feat_frame.configure(width=200)

        # Initialize feature selector with feature importance callback
        self.feature_selector = FeatureSelector(feat_frame, self._on_target_changed, self._on_id_changed, 
                                               self.model_manager.show_feature_importance)

        # Algorithms and hyperparameters side by side
        algo_hp_frame = ttk.Frame(mid)
        algo_hp_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Algorithm frame
        algo_frame = ttk.LabelFrame(algo_hp_frame, text="🤖 Algorithms", style='Modern.TLabelframe')
        algo_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 4))
        algo_frame.configure(width=200)

        # Initialize algorithm selector
        self.algorithm_selector = AlgorithmSelector(algo_frame, self.algos, self._on_algo_configure_click)

        # Hyperparameters frame
        self.hp_frame = ttk.LabelFrame(algo_hp_frame, text="⚙️ Hyperparameters", style='Modern.TLabelframe')
        self.hp_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(4, 0))
        self.hp_frame.configure(width=500)
        
        # Hyperparameters content area
        self.hp_canvas = tk.Canvas(self.hp_frame, borderwidth=0, highlightthickness=0)
        hp_scroll = ttk.Scrollbar(self.hp_frame, orient="vertical", command=self.hp_canvas.yview)
        self.hp_inner = ttk.Frame(self.hp_canvas)
        self.hp_inner.bind(
            "<Configure>", lambda e: self.hp_canvas.configure(scrollregion=self.hp_canvas.bbox("all"))
        )
        self.hp_canvas.create_window((0, 0), window=self.hp_inner, anchor="nw")
        self.hp_canvas.configure(yscrollcommand=hp_scroll.set)
        self.hp_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=4)
        hp_scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=4)

        # Initialize hyperparameter config
        self.hyperparameter_config = HyperparameterConfig(self.hp_inner)
        self.hyperparameter_config.set_algorithms(self.algos)
        # Initialize search strategies for all algorithms
        for algo_name in self.algos.keys():
            self.search_strategies[algo_name] = tk.StringVar(value="Grid Search")
        self.hyperparameter_config.search_strategies = self.search_strategies
        self.hyperparameter_config.show_initial_message()

        # Action buttons
        act = ttk.Frame(self)
        act.pack(fill=tk.X, padx=10, pady=8)
        
        # Main action buttons
        self.run_btn = ttk.Button(act, text="🚀 Run Training", command=self._on_run, style='Primary.TButton')
        self.run_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Data management buttons
        save_btn = ttk.Button(act, text="💾 Save Predictions", command=self._save_predictions, style='Success.TButton')
        save_btn.pack(side=tk.LEFT, padx=5)
        copy_btn = ttk.Button(act, text="📋 Copy Output", command=self._copy_output, style='Warning.TButton')
        copy_btn.pack(side=tk.LEFT, padx=5)
        
        # Prediction and model management buttons
        predict_all_btn = ttk.Button(act, text="🔮 Predict All (Selected)", 
                                   command=self.prediction_manager.predict_all_instances, style='Info.TButton')
        predict_all_btn.pack(side=tk.LEFT, padx=5)
        save_models_btn = ttk.Button(act, text="💾 Save Models", 
                                   command=self.model_manager.save_models, style='Success.TButton')
        save_models_btn.pack(side=tk.LEFT, padx=5)
        load_models_btn = ttk.Button(act, text="📂 Load Models", 
                                   command=self.model_manager.load_models, style='Secondary.TButton')
        load_models_btn.pack(side=tk.LEFT, padx=5)
        
        # Results and charts buttons
        results_btn = ttk.Button(act, text="📊 View Results", command=self._show_results_viewer, style='Info.TButton')
        results_btn.pack(side=tk.RIGHT)
        charts_btn = ttk.Button(act, text="📈 Show Charts", 
                              command=self.charts_manager.show_charts, style='Info.TButton')
        charts_btn.pack(side=tk.RIGHT, padx=(5, 0))

        # Bottom: output + results table
        bottom = ttk.Frame(self)
        bottom.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # Output text with scrollbar
        out = ttk.LabelFrame(bottom, text="📝 Output", style='Modern.TLabelframe')
        out.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        output_frame = ttk.Frame(out)
        output_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        self.output = tk.Text(output_frame, height=16, bg=self.ui_styles.colors['white'], 
                             fg=self.ui_styles.colors['dark'], 
                             font=('Consolas', 9), relief='solid', borderwidth=1)
        output_scroll = ttk.Scrollbar(output_frame, orient="vertical", command=self.output.yview)
        self.output.configure(yscrollcommand=output_scroll.set)
        
        self.output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        output_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # Results table
        table_frame = ttk.LabelFrame(bottom, text="📊 Results Table", style='Modern.TLabelframe')
        table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
        
        # Initialize results table
        self.results_table = ResultsTable(table_frame, self._on_table_click)

        # Try to preload defaults if file exists
        self._try_preload_defaults()

    def _on_target_changed(self):
        """Handle target column change"""
        if self.columns:
            self.feature_selector.rebuild_feature_checks(self.columns, self.target_var.get(), self.id_var.get())

    def _on_id_changed(self):
        """Handle ID column change"""
        if self.columns:
            self.feature_selector.rebuild_feature_checks(self.columns, self.target_var.get(), self.id_var.get())

    def _on_random_state_changed(self):
        """Update random state when changed"""
        try:
            new_random_state = int(self.random_state_var.get())
            self.random_state = new_random_state
            # Update model registry with new random state
            self.algos = get_model_registry(self.random_state)
            # Update algorithm selector
            self.algorithm_selector.update_algorithms(self.algos)
            # Update hyperparameter config
            self.hyperparameter_config.set_algorithms(self.algos)
            # Update search strategies for new algorithms
            for algo_name in self.algos.keys():
                if algo_name not in self.search_strategies:
                    self.search_strategies[algo_name] = tk.StringVar(value="Grid Search")
            self.hyperparameter_config.search_strategies = self.search_strategies
        except (ValueError, TypeError):
            # Invalid random state, keep current value
            pass

    def _on_algo_configure_click(self, algo_name: str):
        """Handle algorithm configure button click"""
        self.hyperparameter_config.show_hyperparameters(algo_name, self.algos)

    def _browse_csv(self):
        """Browse for CSV file"""
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        self.file_var.set(path)
        try:
            df = read_csv(path)
            self.df = df
            self.columns = list_columns(df)
            self.target_cb["values"] = self.columns
            self.id_cb["values"] = self.columns
            self.feature_selector.rebuild_feature_checks(self.columns)
            self._append("Loaded file and updated columns.\n")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV: {e}")
            self._append(f"Error: {e}\n")

    def _try_preload_defaults(self):
        """Try to preload default file"""
        path = self.file_var.get().strip()
        if not path:
            return
        try:
            df = read_csv(path)
            self.df = df
            self.columns = list_columns(df)
            self.target_cb["values"] = self.columns
            self.id_cb["values"] = self.columns
            self.feature_selector.rebuild_feature_checks(self.columns)
            # If default target exists, ensure selection reflects it
            if "group_id" in self.columns:
                self.target_var.set("group_id")
            self._append("Defaults loaded from file.\n")
        except Exception as e:
            self._append(f"Error loading default file: {e}\n")

    def _on_run(self):
        """Run training with selected algorithms"""
        if self.df is None:
            messagebox.showwarning("Missing", "Load a CSV first.")
            return
        target = self.target_var.get().strip()
        if not target:
            messagebox.showwarning("Missing", "Select a target column.")
            return
        features = self.feature_selector.get_selected_features()
        if not features:
            messagebox.showwarning("Missing", "Select at least one feature.")
            return
        # Ensure target and id are not in features (defensive)
        current_id = self.id_var.get().strip() or None
        features = [f for f in features if f != target and (not current_id or f != current_id)]
        id_col = current_id
        try:
            test_size = float(self.test_size_var.get())
        except Exception:
            test_size = 0.3
        selected = self.algorithm_selector.get_selected_algorithms()
        if not selected:
            messagebox.showwarning("Missing", "Select at least one algorithm.")
            return

        self.run_btn.configure(state=tk.DISABLED)
        self._append("Running... This may take a while.\n")
        # Report NaNs prior to preprocessing
        self._report_nans(features, target, id_col)

        def progress_callback(message):
            self._append(message + "\n")
        
        def worker():
            try:
                # Get random state setting first
                try:
                    random_state = int(self.random_state_var.get())
                    self.random_state = random_state
                except (ValueError, TypeError):
                    random_state = 42
                    self.random_state = 42
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=UserWarning)
                    prep = train_test_prepare(
                        df=self.df, target=target, features=features, test_size=test_size,
                        random_state=random_state, stratify=self.stratify_var.get(), id_column=id_col,
                    )
                    # Get custom hyperparameters and search strategies
                    custom_hyperparams = self.hyperparameter_config.get_custom_hyperparameters(selected)
                    search_strategies = {name: self.hyperparameter_config.search_strategies[name].get() for name in selected}
                    
                    # Get n_jobs setting
                    try:
                        n_jobs = int(self.n_jobs_var.get())
                    except (ValueError, TypeError):
                        n_jobs = 8
                    
                    results, best = train_selected(selected, prep, n_jobs=n_jobs, random_state=random_state, 
                                                progress_callback=progress_callback, custom_hyperparams=custom_hyperparams, 
                                                search_strategies=search_strategies)
                
                # Store detailed results for clickable table
                self.detailed_results = best.get("detailed_results", {})
                
                # Store trained models for later use
                self.trained_models = {}
                for name, model_results in self.detailed_results.items():
                    if 'best_estimator' in model_results:
                        self.trained_models[name] = model_results['best_estimator']
                
                # Build predictions for all models
                self.all_predictions = build_all_predictions_df(prep, self.detailed_results)
                
                self._append("\n")
                self._append("Training completed successfully!\n")
                
                # Store results for charts button
                self.last_results_df = results
                
                self.results_table.show_results(results)
                self.charts_manager.draw_charts(results)
                best_name = best.get("best_name")
                best_est = best.get("best_estimator")
                if best_name and best_est is not None:
                    self._append(f"Best model: {best_name}\n")
                    self.predictions_df = build_predictions_df(prep, best_est)
                    self._append("Preview of predictions: \n")
                    self._append(self.predictions_df.head(15).to_string(index=False) + "\n")
                    self._append("\nTip: Double-click any row in the results table to see detailed per-class metrics!\n")
                    self._append("Tip: Click 'View Results' to see individual sample predictions for all models!\n")
                else:
                    self._append("No model produced results.\n")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self._append(f"Error: {e}\n")
            finally:
                self.run_btn.configure(state=tk.NORMAL)

        threading.Thread(target=worker, daemon=True).start()

    def _report_nans(self, features: List[str], target: str, id_col: str | None):
        """Report missing values in selected columns"""
        if self.df is None:
            return
        df = self.df
        cols_to_check = list(features)
        if id_col and id_col in df.columns:
            cols_to_check.append(id_col)
        cols_to_check.append(target)
        nan_counts = df[cols_to_check].isna().sum()
        cols_with_nans = nan_counts[nan_counts > 0]
        
        if len(cols_with_nans) > 0:
            self._append("Missing values found in the following columns:\n")
            for col, count in cols_with_nans.items():
                self._append(f"  • {col}: {count} missing values\n")
            self._append("\nNote: Missing values will be filled with the mean before training.\n")
        else:
            self._append("No missing values found in selected columns.\n")

    def _on_table_click(self, event):
        """Handle double-click on results table"""
        selection = self.results_table.results_table.selection()
        if not selection:
            return
        item = self.results_table.results_table.item(selection[0])
        classifier_name = item['values'][0]
        
        if classifier_name in self.detailed_results:
            self.detailed_metrics.show_detailed_metrics(classifier_name)

    def _show_results_viewer(self):
        """Show detailed results viewer"""
        if not hasattr(self, 'all_predictions') or not self.all_predictions:
            messagebox.showwarning("Missing", "Train models first to view results.")
            return
        
        # Create results viewer window
        results_window = tk.Toplevel(self.master)
        results_window.title("muML - Results Viewer")
        results_window.geometry("1600x800")
        
        # Use ResultsViewer to display results
        ResultsViewer(results_window, self.all_predictions)

    def _save_predictions(self):
        """Save predictions to CSV"""
        if self.predictions_df is None:
            messagebox.showinfo("Info", "Run and produce predictions first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")], initialfile="prediction_results.csv")
        if not path:
            return
        try:
            self.predictions_df.to_csv(path, index=False)
            messagebox.showinfo("Saved", f"Saved to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save: {e}")
            self._append(f"Error: {e}\n")

    def _copy_output(self):
        """Copy output to clipboard"""
        try:
            data = self.output.get("1.0", tk.END)
            self.master.clipboard_clear()
            self.master.clipboard_append(data)
            messagebox.showinfo("Copied", "Output copied to clipboard.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy: {e}")

    def _append(self, text: str):
        """Append text to output"""
        self.output.insert(tk.END, text)
        self.output.see(tk.END)


def main():
    root = tk.Tk()
    # Use ttk theme
    style = ttk.Style()
    try:
        style.theme_use("vista")
    except Exception:
        pass
    app = MLGuiApp(root)
    app.mainloop()


if __name__ == "__main__":
    main()
