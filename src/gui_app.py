from __future__ import annotations

import ast
import threading
import warnings
from typing import Dict, List, Any

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings
warnings.simplefilter("ignore", category=UserWarning)

from src.data import read_csv, list_columns
from src.models import get_model_registry
from src.preprocess import train_test_prepare
from src.train_eval import train_selected, build_predictions_df, build_all_predictions_df

try:
	import shap
	has_shap = True
except ImportError:
	has_shap = False


class MLGuiApp(ttk.Frame):
	def __init__(self, master: tk.Tk):
		super().__init__(master)
		self.master = master
		self.pack(fill=tk.BOTH, expand=True)

		self.df: pd.DataFrame | None = None
		self.columns: List[str] = []
		self.random_state: int = 42
		self.algos = get_model_registry(self.random_state)
		self.selected_algos: Dict[str, tk.BooleanVar] = {k: tk.BooleanVar(value=True) for k in self.algos.keys()}
		self.predictions_df: pd.DataFrame | None = None
		self.detailed_results: Dict[str, Dict[str, Any]] = {}
		self.custom_params: Dict[str, Dict[str, Any]] = {}
		self.search_strategies: Dict[str, tk.StringVar] = {}
		self.all_predictions: Dict[str, pd.DataFrame] = {}

		self.feature_vars: Dict[str, tk.BooleanVar] = {}

		self._build_ui()

	def _build_ui(self):
		self.master.title("muML - Multiple Machine Learning Algorithms")
		self.master.geometry("1600x900")

		# Top controls
		top = ttk.Frame(self)
		top.pack(fill=tk.X, padx=10, pady=8)

		self.file_var = tk.StringVar(value="tree_features.csv")
		file_entry = ttk.Entry(top, textvariable=self.file_var, width=80)
		file_entry.pack(side=tk.LEFT, padx=(0, 8))

		browse_btn = ttk.Button(top, text="Browse CSV", command=self._browse_csv)
		browse_btn.pack(side=tk.LEFT)

		# Options frame
		opts = ttk.LabelFrame(self, text="Options")
		opts.pack(fill=tk.X, padx=10, pady=8)

		self.target_var = tk.StringVar(value="group_id")
		self.target_var.trace_add("write", lambda *args: self._on_target_changed())
		self.id_var = tk.StringVar(value="tree_id")
		self.id_var.trace_add("write", lambda *args: self._on_id_changed())
		self.test_size_var = tk.StringVar(value="0.3")
		self.stratify_var = tk.BooleanVar(value=True)
		self.n_jobs_var = tk.StringVar(value="8")
		self.random_state_var = tk.StringVar(value="42")
		self.random_state_var.trace_add("write", lambda *args: self._on_random_state_changed())

		row1 = ttk.Frame(opts)
		row1.pack(fill=tk.X, padx=8, pady=4)
		ttk.Label(row1, text="Target:").pack(side=tk.LEFT)
		self.target_cb = ttk.Combobox(row1, textvariable=self.target_var, state="readonly", width=30, values=self.columns)
		self.target_cb.pack(side=tk.LEFT, padx=6)

		ttk.Label(row1, text="ID column:").pack(side=tk.LEFT)
		self.id_cb = ttk.Combobox(row1, textvariable=self.id_var, state="readonly", width=20, values=self.columns)
		self.id_cb.pack(side=tk.LEFT, padx=6)

		ttk.Label(row1, text="Test size:").pack(side=tk.LEFT)
		test_entry = ttk.Entry(row1, textvariable=self.test_size_var, width=8)
		test_entry.pack(side=tk.LEFT, padx=6)

		strat_cb = ttk.Checkbutton(row1, text="Stratify", variable=self.stratify_var)
		strat_cb.pack(side=tk.LEFT, padx=6)

		ttk.Label(row1, text="n_jobs:").pack(side=tk.LEFT, padx=(10, 0))
		n_jobs_entry = ttk.Entry(row1, textvariable=self.n_jobs_var, width=8)
		n_jobs_entry.pack(side=tk.LEFT, padx=6)
		ttk.Label(row1, text="(uses 8 cores by default, -1 to use all cores)", font=("Arial", 8), 
				 foreground="gray").pack(side=tk.LEFT, padx=6)
		
		# Random state control
		ttk.Label(row1, text="Random State:").pack(side=tk.LEFT, padx=(20, 0))
		random_state_entry = ttk.Entry(row1, textvariable=self.random_state_var, width=8)
		random_state_entry.pack(side=tk.LEFT, padx=6)
		ttk.Label(row1, text="(controls reproducibility)", font=("Arial", 8), 
				 foreground="gray").pack(side=tk.LEFT, padx=6)

		# Middle: features, algorithms, and hyperparameters
		mid = ttk.Frame(self)
		mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

		feat_frame = ttk.LabelFrame(mid, text="Features")
		feat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 8))
		feat_frame.configure(width=200)

		# Feature control buttons
		feat_ctrl = ttk.Frame(feat_frame)
		feat_ctrl.pack(fill=tk.X, padx=8, pady=4)
		ttk.Button(feat_ctrl, text="Select All", command=self._select_all_features).pack(side=tk.LEFT)
		ttk.Button(feat_ctrl, text="Select None", command=self._select_none_features).pack(side=tk.LEFT, padx=6)
		
		# SHAP feature importance button
		shap_btn = ttk.Button(feat_ctrl, text="Feature Importance", command=self._show_feature_importance)
		shap_btn.pack(side=tk.RIGHT)

		# Scrollable checkbox area
		self.feat_canvas = tk.Canvas(feat_frame, borderwidth=0, highlightthickness=0)
		scroll = ttk.Scrollbar(feat_frame, orient="vertical", command=self.feat_canvas.yview)
		self.feat_inner = ttk.Frame(self.feat_canvas)
		self.feat_inner.bind(
			"<Configure>", lambda e: self.feat_canvas.configure(scrollregion=self.feat_canvas.bbox("all"))
		)
		self.feat_canvas.create_window((0, 0), window=self.feat_inner, anchor="nw")
		self.feat_canvas.configure(yscrollcommand=scroll.set)
		self.feat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=4)
		scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=4)

		# Algorithms and hyperparameters side by side
		algo_hp_frame = ttk.Frame(mid)
		algo_hp_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

		algo_frame = ttk.LabelFrame(algo_hp_frame, text="Algorithms")
		algo_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 4))
		algo_frame.configure(width=200)
		self.algo_checks: Dict[str, ttk.Checkbutton] = {}
		self.algo_buttons: Dict[str, ttk.Button] = {}
		for name, var in self.selected_algos.items():
			# Checkbox for selection
			cb = ttk.Checkbutton(algo_frame, text=name, variable=var)
			cb.pack(anchor=tk.W, padx=8, pady=2)
			self.algo_checks[name] = cb
			
			# Button for hyperparameter configuration
			btn = ttk.Button(algo_frame, text=f"Configure {name}", 
							command=lambda n=name: self._on_algo_configure_click(n))
			btn.pack(anchor=tk.W, padx=8, pady=1)
			self.algo_buttons[name] = btn

		# Hyperparameters frame
		self.hp_frame = ttk.LabelFrame(algo_hp_frame, text="Hyperparameters")
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

		# Action buttons
		act = ttk.Frame(self)
		act.pack(fill=tk.X, padx=10, pady=8)
		self.run_btn = ttk.Button(act, text="Run", command=self._on_run)
		self.run_btn.pack(side=tk.LEFT)
		save_btn = ttk.Button(act, text="Save predictions CSV", command=self._save_predictions)
		save_btn.pack(side=tk.LEFT, padx=8)
		copy_btn = ttk.Button(act, text="Copy Output", command=self._copy_output)
		copy_btn.pack(side=tk.LEFT, padx=8)
		results_btn = ttk.Button(act, text="View Results", command=self._show_results_viewer)
		results_btn.pack(side=tk.RIGHT)

		# Bottom: output + results table
		bottom = ttk.Frame(self)
		bottom.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

		# Output text with scrollbar
		out = ttk.LabelFrame(bottom, text="Output")
		out.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		
		output_frame = ttk.Frame(out)
		output_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
		
		self.output = tk.Text(output_frame, height=16)
		output_scroll = ttk.Scrollbar(output_frame, orient="vertical", command=self.output.yview)
		self.output.configure(yscrollcommand=output_scroll.set)
		
		self.output.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		output_scroll.pack(side=tk.RIGHT, fill=tk.Y)

		# Results table
		table_frame = ttk.LabelFrame(bottom, text="Results Table")
		table_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0))
		
		table_container = ttk.Frame(table_frame)
		table_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
		
		self.results_table = ttk.Treeview(table_container, columns=("Classifier", "Accuracy", "F1-Score"), show="headings")
		self.results_table.heading("Classifier", text="Classifier")
		self.results_table.heading("Accuracy", text="Accuracy")
		self.results_table.heading("F1-Score", text="F1-Score")
		self.results_table.column("Classifier", width=180, anchor=tk.W)
		self.results_table.column("Accuracy", width=90, anchor=tk.CENTER)
		self.results_table.column("F1-Score", width=90, anchor=tk.CENTER)
		
		table_scroll = ttk.Scrollbar(table_container, orient="vertical", command=self.results_table.yview)
		self.results_table.configure(yscrollcommand=table_scroll.set)
		
		self.results_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		table_scroll.pack(side=tk.RIGHT, fill=tk.Y)
		
		# Bind click event to results table
		self.results_table.bind("<Double-1>", self._on_table_click)

		# Initialize hyperparameter widgets
		self.hp_widgets: Dict[str, Dict[str, Any]] = {}
		self._init_hyperparameter_widgets()
		
		# Show initial hyperparameters message
		self._show_initial_hyperparameters_message()

		# Try to preload defaults if file exists
		self._try_preload_defaults()

	def _copy_output(self):
		try:
			data = self.output.get("1.0", tk.END)
			self.master.clipboard_clear()
			self.master.clipboard_append(data)
			messagebox.showinfo("Copied", "Output copied to clipboard.")
		except Exception as e:
			messagebox.showerror("Error", f"Failed to copy: {e}")

	def _try_preload_defaults(self):
		path = self.file_var.get().strip()
		if not path:
			return
		try:
			df = read_csv(path)
			self.df = df
			self.columns = list_columns(df)
			self.target_cb["values"] = self.columns
			self.id_cb["values"] = self.columns
			self._rebuild_feature_checks(self.columns)
			# If default target exists, ensure selection reflects it
			if "group_id" in self.columns:
				self.target_var.set("group_id")
			self._append("Defaults loaded from file.\n")
		except Exception as e:
			self._append(f"Error loading default file: {e}\n")

	def _rebuild_feature_checks(self, cols: List[str]):
		# Clear existing
		for child in list(self.feat_inner.children.values()):
			child.destroy()
		self.feature_vars.clear()
		# Create a checkbutton for each column (skip target and id)
		current_target = self.target_var.get().strip()
		current_id = self.id_var.get().strip()
		for c in cols:
			if (current_target and c == current_target) or (current_id and c == current_id):
				continue
			var = tk.BooleanVar(value=True)
			self.feature_vars[c] = var
			cb = ttk.Checkbutton(self.feat_inner, text=c, variable=var)
			cb.pack(anchor=tk.W, padx=4, pady=2)

	def _select_all_features(self):
		for name, var in self.feature_vars.items():
			var.set(True)

	def _select_none_features(self):
		for var in self.feature_vars.values():
			var.set(False)

	def _on_target_changed(self):
		# Rebuild features to ensure target is excluded
		if self.columns:
			self._rebuild_feature_checks(self.columns)

	def _on_id_changed(self):
		# Rebuild features to ensure id is excluded
		if self.columns:
			self._rebuild_feature_checks(self.columns)

	def _on_random_state_changed(self):
		"""Update random state when changed"""
		try:
			new_random_state = int(self.random_state_var.get())
			self.random_state = new_random_state
			# Update model registry with new random state
			self.algos = get_model_registry(self.random_state)
		except (ValueError, TypeError):
			# Invalid random state, keep current value
			pass

	def _init_hyperparameter_widgets(self):
		"""Initialize hyperparameter widgets for all algorithms"""
		for algo_name in self.algos.keys():
			self.hp_widgets[algo_name] = {}
			self.search_strategies[algo_name] = tk.StringVar(value="Grid Search")
			self.custom_params[algo_name] = {}

	def _on_algo_configure_click(self, algo_name: str):
		"""Handle algorithm configure button click to show hyperparameters"""
		self._show_hyperparameters(algo_name)

	def _show_hyperparameters(self, algo_name: str):
		"""Show hyperparameter controls for selected algorithm"""
		if algo_name not in self.algos:
			return
		
		# Clear existing hyperparameter widgets
		for child in list(self.hp_inner.children.values()):
			child.destroy()
		
		# Clear the widget references for this algorithm to prevent stale references
		if algo_name in self.hp_widgets:
			del self.hp_widgets[algo_name]
		
		algo_info = self.algos[algo_name]
		params = algo_info.get("params", {})
		
		if not params:
			ttk.Label(self.hp_inner, text=f"No hyperparameters for {algo_name}", 
					 font=("Arial", 10, "italic")).pack(pady=10)
			return
		
		# Search strategy selection
		strategy_frame = ttk.Frame(self.hp_inner)
		strategy_frame.pack(fill=tk.X, pady=(0, 10))
		ttk.Label(strategy_frame, text="Search Strategy:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
		ttk.Radiobutton(strategy_frame, text="Grid Search", variable=self.search_strategies[algo_name], 
					   value="Grid Search").pack(side=tk.LEFT, padx=(10, 5))
		ttk.Radiobutton(strategy_frame, text="Random Search", variable=self.search_strategies[algo_name], 
					   value="Random Search").pack(side=tk.LEFT, padx=5)
		
		# Hyperparameter controls
		ttk.Label(self.hp_inner, text=f"Hyperparameters for {algo_name}:", 
				 font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
		
		# Initialize fresh widget dictionary for this algorithm
		self.hp_widgets[algo_name] = {}
		
		for param_name, default_values in params.items():
			param_frame = ttk.Frame(self.hp_inner)
			param_frame.pack(fill=tk.X, pady=2)
			
			ttk.Label(param_frame, text=f"{param_name}:", width=25, anchor=tk.W).pack(side=tk.LEFT)
			
			# Create entry widget for parameter values
			entry = ttk.Entry(param_frame, width=40)
			entry.pack(side=tk.LEFT, padx=(5, 0))
			
			# Set default values
			if isinstance(default_values, list):
				entry.insert(0, str(default_values))
			else:
				entry.insert(0, str(default_values))
			
			# Store reference
			self.hp_widgets[algo_name][param_name] = entry
			
			# Add help text
			help_text = f"Default: {default_values}"
			ttk.Label(param_frame, text=help_text, font=("Arial", 8), 
					 foreground="gray").pack(side=tk.LEFT, padx=(10, 0))

	def _show_initial_hyperparameters_message(self):
		"""Show initial message in hyperparameters frame"""
		# Clear hyperparameter area
		for child in list(self.hp_inner.children.values()):
			child.destroy()
		
		# Clear all widget references to prevent stale references
		self.hp_widgets.clear()
		
		ttk.Label(self.hp_inner, text="Click 'Configure [Algorithm]' to set hyperparameters", 
				 font=("Arial", 10, "italic")).pack(pady=20)
		
		# Add instruction text
		instruction_text = """Instructions:
• Click "Configure [Algorithm]" to set hyperparameters
• Enter values as lists: [1, 2, 3] or single values: 5
• Choose Grid Search for exhaustive search or Random Search for sampling
• Leave empty to use default values
• Check algorithms to include them in training
• Adjust n_jobs to control CPU usage (8 = default, -1 = all cores, 1 = single core)"""
		ttk.Label(self.hp_inner, text=instruction_text, font=("Arial", 8), 
				 foreground="gray", justify=tk.LEFT).pack(pady=10, padx=10)

	def _hide_hyperparameters(self, algo_name: str):
		"""Hide hyperparameter controls for deselected algorithm"""
		self._show_initial_hyperparameters_message()

	def _get_custom_hyperparameters(self) -> Dict[str, Dict[str, Any]]:
		"""Get custom hyperparameters from UI"""
		custom_params = {}
		
		# Only get hyperparameters for currently selected algorithms
		selected_algos = [name for name, var in self.selected_algos.items() if var.get()]
		
		for algo_name in selected_algos:
			if algo_name not in self.hp_widgets:
				continue
				
			algo_params = {}
			widgets = self.hp_widgets[algo_name]
			
			for param_name, entry_widget in widgets.items():
				try:
					# Check if widget still exists and is valid
					if not hasattr(entry_widget, 'get'):
						continue
					
					# Try to get the value, but handle widget destruction gracefully
					try:
						value_str = entry_widget.get().strip()
					except tk.TclError:
						# Widget has been destroyed, skip it
						continue
						
					if value_str:
						# Try to parse as list first, then single value
						if value_str.startswith('[') and value_str.endswith(']'):
							# Parse as list
							algo_params[param_name] = ast.literal_eval(value_str)
						else:
							# Try to parse as single value
							try:
								# Try integer
								algo_params[param_name] = int(value_str)
							except ValueError:
								try:
									# Try float
									algo_params[param_name] = float(value_str)
								except ValueError:
									# Keep as string
									algo_params[param_name] = value_str
				except Exception as e:
					# Only show warning if it's not a widget destruction error
					if "invalid command name" not in str(e):
						self._append(f"Warning: Invalid parameter value for {algo_name}.{param_name}: {e}\n")
					continue
			
			if algo_params:
				custom_params[algo_name] = algo_params
		
		return custom_params

	def _browse_csv(self):
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
			self._rebuild_feature_checks(self.columns)
			self._append("Loaded file and updated columns.\n")
		except Exception as e:
			messagebox.showerror("Error", f"Failed to load CSV: {e}")
			self._append(f"Error: {e}\n")

	def _collect_selected_features(self) -> List[str]:
		selected: List[str] = []
		for name, var in self.feature_vars.items():
			if var.get():
				selected.append(name)
		return selected

	def _report_nans(self, features: List[str], target: str, id_col: str | None):
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


	def _clear_results_table(self):
		for row in self.results_table.get_children():
			self.results_table.delete(row)

	def _show_results_table(self, results_df: pd.DataFrame):
		self._clear_results_table()
		for _, r in results_df.iterrows():
			sclf = str(r.get("Classifier"))
			acc = f"{float(r.get('Accuracy', 0.0)):.4f}"
			f1 = f"{float(r.get('F1-Score', 0.0)):.4f}"
			self.results_table.insert("", tk.END, values=(sclf, acc, f1))

	def _on_table_click(self, event):
		selection = self.results_table.selection()
		if not selection:
			return
		item = self.results_table.item(selection[0])
		classifier_name = item['values'][0]
		
		if classifier_name in self.detailed_results:
			self._show_detailed_metrics(classifier_name)

	def _show_detailed_metrics(self, classifier_name: str):
		if classifier_name not in self.detailed_results:
			return
		
		details = self.detailed_results[classifier_name]
		crep = details["classification_report"]
		
		# Create popup window for detailed metrics
		detail_window = tk.Toplevel(self.master)
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

	def _draw_charts(self, results_df: pd.DataFrame):
		if results_df.empty:
			return
		# Create popup window for charts
		chart_window = tk.Toplevel(self.master)
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

	def _on_run(self):
		if self.df is None:
			messagebox.showwarning("Missing", "Load a CSV first.")
			return
		target = self.target_var.get().strip()
		if not target:
			messagebox.showwarning("Missing", "Select a target column.")
			return
		features = self._collect_selected_features()
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
		selected = [k for k, v in self.selected_algos.items() if v.get()]
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
					warnings.simplefilter("ignore", category=ConvergenceWarning)
					warnings.simplefilter("ignore", category=FutureWarning)
					warnings.simplefilter("ignore", category=UserWarning)
					prep = train_test_prepare(
						df=self.df, target=target, features=features, test_size=test_size,
						random_state=random_state, stratify=self.stratify_var.get(), id_column=id_col,
					)
					# Get custom hyperparameters and search strategies
					custom_hyperparams = self._get_custom_hyperparameters()
					search_strategies = {name: self.search_strategies[name].get() for name in selected}
					
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
				
				# Build predictions for all models
				self.all_predictions = build_all_predictions_df(prep, self.detailed_results)
				
				self._append("\n")
				self._append("Training completed successfully!\n")
				self._show_results_table(results)
				self._draw_charts(results)
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

	def _save_predictions(self):
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

	def _append(self, text: str):
		self.output.insert(tk.END, text)
		self.output.see(tk.END)

	def _show_feature_importance(self):
		"""Show SHAP feature importance analysis for top 4 models"""
		if not has_shap:
			messagebox.showerror("Error", "SHAP is not installed. Please install it with: pip install shap")
			return
		
		if self.df is None:
			messagebox.showwarning("Missing", "Load a CSV file first.")
			return
		
		# Get selected features
		features = self._collect_selected_features()
		if not features:
			messagebox.showwarning("Missing", "Select at least one feature.")
			return
		
		target = self.target_var.get().strip()
		if not target:
			messagebox.showwarning("Missing", "Select a target column.")
			return
		
		# Check if we have a trained model
		if not hasattr(self, 'detailed_results') or not self.detailed_results:
			messagebox.showwarning("Missing", "Train a model first to analyze feature importance.")
			return
		
		# Check if we have any valid models for SHAP analysis
		valid_models = ["Random Forest", "Logistic Regression", "SVM", "Decision Tree", "AdaBoost", "KNN", "Naive Bayes", "XGBoost"]
		available_models = [name for name in self.detailed_results.keys() if name in valid_models]
		
		if not available_models:
			messagebox.showwarning("Missing", "No models available for SHAP analysis. Train some models first.")
			return
		
		# Get top 4 models by F1 score
		model_scores = []
		for name in available_models:
			if name in self.detailed_results:
				results = self.detailed_results[name]
				f1_score = results.get('f1_weighted', 0)
				model_scores.append((name, f1_score))
		
		# Sort by F1 score and take top 4
		model_scores.sort(key=lambda x: x[1], reverse=True)
		top_models = [name for name, score in model_scores[:4]]
		
		if not top_models:
			messagebox.showerror("Error", "No trained models found for feature importance analysis.")
			return
		
		# Show loading message
		self._append(f"Calculating SHAP feature importance for top {len(top_models)} models...\n")
		
		# Run SHAP analysis in a separate thread
		def shap_worker():
			try:
				import numpy as np
				import threading
				# Prepare data
				X = self.df[features]
				y = self.df[target]
				
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
						self.master.after(0, lambda name=model_name: self._append(f"SHAP analysis not supported for {name}\n"))
						continue
					
					try:
						# Train a simple version of the model
						model_class = model_classes[model_name]
						
						# Special handling for different models
						if model_name == "XGBoost":
							model = model_class(random_state=self.random_state, eval_metric="mlogloss")
						elif model_name == "KNN":
							# KNN doesn't support random_state
							model = model_class()
						else:
							model = model_class(random_state=self.random_state)
						
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
								model = SVC(random_state=self.random_state, probability=True)
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
							self.master.after(0, lambda name=model_name: self._append(f"Generated SHAP values for {name}\n"))
						else:
							self.master.after(0, lambda name=model_name: self._append(f"Failed to generate SHAP values for {name}\n"))
					
					except Exception as model_error:
						self.master.after(0, lambda name=model_name, err=model_error: self._append(f"Error with {name}: {err}\n"))
						continue
				
				# Show SHAP plots for all successful models
				if shap_results:
					self.master.after(0, lambda: self._display_multiple_shap_plots(shap_results, features))
				else:
					self.master.after(0, lambda: self._append("No SHAP values could be generated for any model\n"))
				
			except Exception as e:
				error_msg = f"SHAP analysis error: {e}\n"
				self.master.after(0, lambda msg=error_msg: self._append(msg))
		
		threading.Thread(target=shap_worker, daemon=True).start()

	def _display_multiple_shap_plots(self, shap_results, features):
		"""Display SHAP feature importance as colorful bar charts for multiple models"""
		try:
			# Create popup window for SHAP plots
			shap_window = tk.Toplevel(self.master)
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
			
			self._append(f"SHAP feature importance analysis completed for {len(shap_results)} models\n")
			
		except Exception as e:
			messagebox.showerror("Error", f"Failed to display SHAP plots: {e}")
			self._append(f"SHAP display error: {e}\n")


	def _show_results_viewer(self):
		"""Show detailed results viewer with models as columns for easy comparison"""
		if not hasattr(self, 'all_predictions') or not self.all_predictions:
			messagebox.showwarning("Missing", "Train models first to view results.")
			return
		
		# Create results viewer window
		results_window = tk.Toplevel(self.master)
		results_window.title("muML - Results Viewer")
		results_window.geometry("1600x800")
		
		# Main frame
		main_frame = ttk.Frame(results_window)
		main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
		
		# Title
		title_label = ttk.Label(main_frame, text="Individual Sample Results - Model Comparison", 
							   font=("Arial", 14, "bold"))
		title_label.pack(pady=(0, 10))
		
		# Filter controls frame
		filter_frame = ttk.LabelFrame(main_frame, text="Filters")
		filter_frame.pack(fill=tk.X, pady=(0, 10))
		
		# Filter controls
		filter_controls = ttk.Frame(filter_frame)
		filter_controls.pack(fill=tk.X, padx=10, pady=10)
		
		# Split filter
		ttk.Label(filter_controls, text="Split:").pack(side=tk.LEFT)
		split_var = tk.StringVar(value="All")
		split_cb = ttk.Combobox(filter_controls, textvariable=split_var, state="readonly", width=15)
		split_cb['values'] = ["All", "train", "test"]
		split_cb.pack(side=tk.LEFT, padx=(5, 20))
		
		# Group filter
		ttk.Label(filter_controls, text="Group:").pack(side=tk.LEFT)
		group_var = tk.StringVar(value="All Groups")
		group_cb = ttk.Combobox(filter_controls, textvariable=group_var, state="readonly", width=15)
		
		# Get unique groups from all predictions
		all_groups = set()
		for df in self.all_predictions.values():
			all_groups.update(df['actual_group'].unique())
		group_cb['values'] = ["All Groups"] + sorted([str(g) for g in all_groups])
		group_cb.pack(side=tk.LEFT, padx=(5, 20))
		
		# Correctness filter
		ttk.Label(filter_controls, text="Correctness:").pack(side=tk.LEFT)
		correct_var = tk.StringVar(value="All")
		correct_cb = ttk.Combobox(filter_controls, textvariable=correct_var, state="readonly", width=15)
		correct_cb['values'] = ["All", "Correct", "Incorrect"]
		correct_cb.pack(side=tk.LEFT, padx=(5, 20))
		
		# Refresh button
		refresh_btn = ttk.Button(filter_controls, text="Refresh", 
								command=lambda: self._refresh_results_view(results_tree, split_var, group_var, correct_var, accuracy_frame))
		refresh_btn.pack(side=tk.RIGHT)
		
		# Results tree frame
		tree_frame = ttk.Frame(main_frame)
		tree_frame.pack(fill=tk.BOTH, expand=True)
		
		# Create treeview for results with models as columns
		model_names = list(self.all_predictions.keys())
		columns = ["tree_id", "actual_group", "split"] + model_names
		results_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=20)
		
		# Configure columns
		results_tree.heading("tree_id", text="ID")
		results_tree.heading("actual_group", text="Actual")
		results_tree.heading("split", text="Split")
		
		# Configure model columns
		for model_name in model_names:
			results_tree.heading(model_name, text=model_name)
		
		# Set column widths
		results_tree.column("tree_id", width=60, anchor=tk.CENTER)
		results_tree.column("actual_group", width=80, anchor=tk.CENTER)
		results_tree.column("split", width=60, anchor=tk.CENTER)
		
		# Set model column widths
		for model_name in model_names:
			results_tree.column(model_name, width=100, anchor=tk.CENTER)
		
		# Add scrollbars
		tree_scroll_v = ttk.Scrollbar(tree_frame, orient="vertical", command=results_tree.yview)
		tree_scroll_h = ttk.Scrollbar(tree_frame, orient="horizontal", command=results_tree.xview)
		results_tree.configure(yscrollcommand=tree_scroll_v.set, xscrollcommand=tree_scroll_h.set)
		
		# Pack treeview and scrollbars
		results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		tree_scroll_v.pack(side=tk.RIGHT, fill=tk.Y)
		tree_scroll_h.pack(side=tk.BOTTOM, fill=tk.X)
		
		# Model accuracy display frame
		accuracy_frame = ttk.LabelFrame(main_frame, text="Model Accuracies (Filtered Results)")
		accuracy_frame.pack(fill=tk.X, pady=(10, 5))
		
		# Create accuracy display
		accuracy_display = ttk.Frame(accuracy_frame)
		accuracy_display.pack(fill=tk.X, padx=10, pady=5)
		
		
		# Status bar
		status_frame = ttk.Frame(main_frame)
		status_frame.pack(fill=tk.X, pady=(10, 0))
		self.status_label = ttk.Label(status_frame, text="Ready")
		self.status_label.pack(side=tk.LEFT)
		
		# Load initial data
		self._refresh_results_view(results_tree, split_var, group_var, correct_var, accuracy_frame)

	def _refresh_results_view(self, tree, split_var, group_var, correct_var, accuracy_frame):
		"""Refresh the results tree view based on filters with models as columns"""
		# Clear existing items
		for item in tree.get_children():
			tree.delete(item)
		
		# Get filter values
		selected_split = split_var.get()
		selected_group = group_var.get()
		selected_correct = correct_var.get()
		
		# Get model names
		model_names = list(self.all_predictions.keys())
		
		# Get all unique tree IDs from all models
		all_tree_ids = set()
		for df in self.all_predictions.values():
			all_tree_ids.update(df['tree_id'].unique())
		
		# Create a combined dataframe for each tree_id
		combined_data = []
		for tree_id in sorted(all_tree_ids):
			# Get data for this tree_id from all models
			tree_data = {'tree_id': tree_id}
			
			# Get actual group and split from first model (should be same for all)
			first_model_data = next(iter(self.all_predictions.values()))
			tree_row = first_model_data[first_model_data['tree_id'] == tree_id]
			if tree_row.empty:
				continue
			
			tree_data['actual_group'] = tree_row.iloc[0]['actual_group']
			tree_data['split'] = tree_row.iloc[0]['split']
			
			# Apply filters
			if selected_split != "All" and tree_data['split'] != selected_split:
				continue
			
			if selected_group != "All Groups" and str(tree_data['actual_group']) != selected_group:
				continue
			
			# Get predictions from all models
			all_correct = True
			all_incorrect = True
			for model_name in model_names:
				model_df = self.all_predictions[model_name]
				model_row = model_df[model_df['tree_id'] == tree_id]
				
				if not model_row.empty:
					pred = model_row.iloc[0]['predicted_group']
					is_correct = model_row.iloc[0]['is_correct']
					
					if pd.notna(pred):
						# Add enhanced visual indicators for correct/incorrect predictions
						if is_correct:
							tree_data[model_name] = f"✅ {pred}"
							all_incorrect = False
						else:
							tree_data[model_name] = f"❌ {pred}"
							all_correct = False
					else:
						tree_data[model_name] = "➖ N/A"
						all_correct = False
						all_incorrect = False
				else:
					tree_data[model_name] = "N/A"
					all_correct = False
					all_incorrect = False
			
			# Apply correctness filter
			if selected_correct == "Correct" and not all_correct:
				continue
			elif selected_correct == "Incorrect" and not all_incorrect:
				continue
			
			# Add to combined data
			row_values = [tree_data['tree_id'], tree_data['actual_group'], tree_data['split']]
			for model_name in model_names:
				row_values.append(tree_data.get(model_name, "N/A"))
			
			combined_data.append(row_values)
		
		# Insert data into tree with enhanced visual indicators
		for data in combined_data:
			# Insert row
			item = tree.insert("", tk.END, values=data)
		
		# Calculate and display per-model accuracies
		self._update_model_accuracies(combined_data, model_names, accuracy_frame)
		
		# Update status
		total_samples = len(combined_data)
		self.status_label.config(text=f"Showing {total_samples} samples across {len(model_names)} models")

	def _update_model_accuracies(self, combined_data, model_names, accuracy_frame):
		"""Update the model accuracy display based on filtered data"""
		# Clear existing accuracy labels
		for widget in accuracy_frame.winfo_children():
			if isinstance(widget, ttk.Frame):
				widget.destroy()
		
		# Create new accuracy display
		accuracy_display = ttk.Frame(accuracy_frame)
		accuracy_display.pack(fill=tk.X, padx=10, pady=5)
		
		# Calculate accuracies for each model
		model_accuracies = {}
		total_samples = len(combined_data)
		
		if total_samples == 0:
			ttk.Label(accuracy_display, text="No samples match the current filters", 
					 font=("Arial", 10, "italic")).pack()
			return
		
		for model_name in model_names:
			correct_count = 0
			total_predictions = 0
			
			for data in combined_data:
				# Get prediction for this model
				col_index = 3 + model_names.index(model_name)  # Skip ID, actual_group, split columns
				prediction = data[col_index]
				
				if prediction != "N/A" and not prediction.startswith("➖"):
					total_predictions += 1
					# Check if prediction is correct
					if prediction.startswith("✅"):
						correct_count += 1
					elif not prediction.startswith("❌"):
						# Fallback: check if prediction matches actual
						actual_group = data[1]
						if str(prediction) == str(actual_group):
							correct_count += 1
			
			# Calculate accuracy
			accuracy = (correct_count / total_predictions * 100) if total_predictions > 0 else 0
			model_accuracies[model_name] = {
				'accuracy': accuracy,
				'correct': correct_count,
				'total': total_predictions
			}
		
		# Sort models by accuracy (descending)
		sorted_models = sorted(model_accuracies.items(), key=lambda x: x[1]['accuracy'], reverse=True)
		
		# Display accuracies in a single row - fit all models
		for col, (model_name, stats) in enumerate(sorted_models):
			# Create model accuracy frame
			model_frame = ttk.Frame(accuracy_display)
			model_frame.grid(row=0, column=col, padx=4, pady=3, sticky="ew")
			
			# Model name (shorter for space)
			name_label = ttk.Label(model_frame, text=model_name, font=("Arial", 8, "bold"))
			name_label.pack()
			
			# Accuracy percentage
			acc_label = ttk.Label(model_frame, text=f"{stats['accuracy']:.1f}%", 
								font=("Arial", 10, "bold"), foreground="blue")
			acc_label.pack()
			
			# Correct/Total count (smaller)
			count_label = ttk.Label(model_frame, text=f"({stats['correct']}/{stats['total']})", 
								  font=("Arial", 7), foreground="gray")
			count_label.pack()
		
		# Configure grid weights for all columns
		for i in range(len(sorted_models)):
			accuracy_display.columnconfigure(i, weight=1)


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
