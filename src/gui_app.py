from __future__ import annotations

import ast
import threading
import warnings
from typing import Dict, List, Any

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.exceptions import ConvergenceWarning

from src.data import read_csv, list_columns
from src.models import get_model_registry
from src.preprocess import train_test_prepare
from src.train_eval import train_selected, build_predictions_df


class MLGuiApp(ttk.Frame):
	def __init__(self, master: tk.Tk):
		super().__init__(master)
		self.master = master
		self.pack(fill=tk.BOTH, expand=True)

		self.df: pd.DataFrame | None = None
		self.columns: List[str] = []
		self.algos = get_model_registry()
		self.selected_algos: Dict[str, tk.BooleanVar] = {k: tk.BooleanVar(value=True) for k in self.algos.keys()}
		self.predictions_df: pd.DataFrame | None = None
		self.detailed_results: Dict[str, Dict[str, Any]] = {}
		self.custom_params: Dict[str, Dict[str, Any]] = {}
		self.search_strategies: Dict[str, tk.StringVar] = {}

		self.feature_vars: Dict[str, tk.BooleanVar] = {}

		self._build_ui()

	def _build_ui(self):
		self.master.title("muML - Multiple Machine Learning Algorithms")
		self.master.geometry("1400x800")

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
		copy_btn.pack(side=tk.LEFT)

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
		detail_window.geometry("1000x700")
		
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
		content += f"Overall Accuracy: {details['accuracy']:.4f}\n"
		content += f"F1-Score (Weighted): {details['f1_weighted']:.4f}\n"
		content += f"Training Time: {details['training_time']:.2f} seconds\n\n"
		
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
		chart_window.geometry("1000x600")
		
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
				with warnings.catch_warnings():
					warnings.simplefilter("ignore", category=ConvergenceWarning)
					warnings.simplefilter("ignore", category=FutureWarning)
					prep = train_test_prepare(
						df=self.df, target=target, features=features, test_size=test_size,
						stratify=self.stratify_var.get(), id_column=id_col,
					)
					# Get custom hyperparameters and search strategies
					custom_hyperparams = self._get_custom_hyperparameters()
					search_strategies = {name: self.search_strategies[name].get() for name in selected}
					
					# Get n_jobs setting
					try:
						n_jobs = int(self.n_jobs_var.get())
					except (ValueError, TypeError):
						n_jobs = 8
					
					results, best = train_selected(selected, prep, n_jobs=n_jobs, progress_callback=progress_callback,
												custom_hyperparams=custom_hyperparams, search_strategies=search_strategies)
				
				# Store detailed results for clickable table
				self.detailed_results = best.get("detailed_results", {})
				
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
