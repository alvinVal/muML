from __future__ import annotations

import threading
from typing import Dict, List

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
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

		self.feature_vars: Dict[str, tk.BooleanVar] = {}

		self._build_ui()

	def _build_ui(self):
		self.master.title("muML - Multiple Machine Learning Algorithms")
		self.master.geometry("1100x700")

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

		# Middle: features and algorithms
		mid = ttk.Frame(self)
		mid.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

		feat_frame = ttk.LabelFrame(mid, text="Features")
		feat_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

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

		algo_frame = ttk.LabelFrame(mid, text="Algorithms")
		algo_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		self.algo_checks: Dict[str, ttk.Checkbutton] = {}
		for name, var in self.selected_algos.items():
			cb = ttk.Checkbutton(algo_frame, text=name, variable=var)
			cb.pack(anchor=tk.W, padx=8, pady=2)
			self.algo_checks[name] = cb

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
		detail_window.geometry("600x500")
		
		# Create scrollable text area
		text_frame = ttk.Frame(detail_window)
		text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
		
		text_widget = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10))
		scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=text_widget.yview)
		text_widget.configure(yscrollcommand=scrollbar.set)
		
		text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
		scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
		
		# Format detailed metrics
		content = f"Detailed Metrics for {classifier_name}\n"
		content += "=" * 50 + "\n\n"
		content += f"Overall Accuracy: {details['accuracy']:.4f}\n"
		content += f"F1-Score (Weighted): {details['f1_weighted']:.4f}\n"
		content += f"Training Time: {details['training_time']:.2f} seconds\n\n"
		
		content += "Per-Class Metrics:\n"
		content += "-" * 30 + "\n"
		content += f"{'Class':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}\n"
		content += "-" * 30 + "\n"
		
		# Show per-class metrics
		for class_name in sorted(crep.keys()):
			if class_name in ['accuracy', 'macro avg', 'weighted avg']:
				continue
			metrics = crep[class_name]
			content += f"{class_name:<8} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1-score']:<10.4f} {metrics['support']:<8.0f}\n"
		
		# Add summary metrics
		content += "\nSummary:\n"
		content += "-" * 30 + "\n"
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
					results, best = train_selected(selected, prep, n_jobs=-1, progress_callback=progress_callback)
				
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

	def _clear(self):
		self.output.delete("1.0", tk.END)


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
