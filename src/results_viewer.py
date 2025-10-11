"""
Results Viewer module for muML
Handles results display, filtering, and visualization
"""

import tkinter as tk
from tkinter import ttk
import pandas as pd
from typing import Dict, List, Any, Optional


class ResultsViewer:
    """Handles the main results viewer with filtering and model comparison"""
    
    def __init__(self, parent_window, all_predictions: Dict[str, pd.DataFrame]):
        self.parent_window = parent_window
        self.all_predictions = all_predictions
        self.status_label = None
        self._build_ui()
    
    def _build_ui(self):
        """Build the results viewer UI"""
        # Main frame
        main_frame = ttk.Frame(self.parent_window)
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
        self.split_var = tk.StringVar(value="All")
        split_cb = ttk.Combobox(filter_controls, textvariable=self.split_var, state="readonly", width=15)
        split_cb['values'] = ["All", "train", "test"]
        split_cb.pack(side=tk.LEFT, padx=(5, 20))
        
        # Group filter
        ttk.Label(filter_controls, text="Group:").pack(side=tk.LEFT)
        self.group_var = tk.StringVar(value="All Groups")
        group_cb = ttk.Combobox(filter_controls, textvariable=self.group_var, state="readonly", width=15)
        
        # Get unique groups from all predictions
        all_groups = set()
        for df in self.all_predictions.values():
            all_groups.update(df['actual_group'].unique())
        group_cb['values'] = ["All Groups"] + sorted([str(g) for g in all_groups])
        group_cb.pack(side=tk.LEFT, padx=(5, 20))
        
        # Correctness filter
        ttk.Label(filter_controls, text="Correctness:").pack(side=tk.LEFT)
        self.correct_var = tk.StringVar(value="All")
        correct_cb = ttk.Combobox(filter_controls, textvariable=self.correct_var, state="readonly", width=15)
        correct_cb['values'] = ["All", "Correct", "Incorrect"]
        correct_cb.pack(side=tk.LEFT, padx=(5, 20))
        
        # Refresh button
        refresh_btn = ttk.Button(filter_controls, text="Refresh", 
                                command=self._refresh_view)
        refresh_btn.pack(side=tk.RIGHT)
        
        # Results tree frame
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for results with models as columns
        model_names = list(self.all_predictions.keys())
        columns = ["tree_id", "actual_group", "split"] + model_names
        self.results_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=20)
        
        # Configure columns
        self.results_tree.heading("tree_id", text="ID")
        self.results_tree.heading("actual_group", text="Actual")
        self.results_tree.heading("split", text="Split")
        
        # Configure model columns
        for model_name in model_names:
            self.results_tree.heading(model_name, text=model_name)
        
        # Set column widths
        self.results_tree.column("tree_id", width=60, anchor=tk.CENTER)
        self.results_tree.column("actual_group", width=80, anchor=tk.CENTER)
        self.results_tree.column("split", width=60, anchor=tk.CENTER)
        
        # Set model column widths
        for model_name in model_names:
            self.results_tree.column(model_name, width=100, anchor=tk.CENTER)
        
        # Add scrollbars
        tree_scroll_v = ttk.Scrollbar(tree_frame, orient="vertical", command=self.results_tree.yview)
        tree_scroll_h = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=tree_scroll_v.set, xscrollcommand=tree_scroll_h.set)
        
        # Pack treeview and scrollbars
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll_v.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_h.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Color legend frame
        legend_frame = ttk.LabelFrame(main_frame, text="Color Legend")
        legend_frame.pack(fill=tk.X, pady=(5, 10))
        
        # Create legend with color indicators
        legend_controls = ttk.Frame(legend_frame)
        legend_controls.pack(fill=tk.X, padx=10, pady=5)
        
        # Legend items
        legend_items = [
            ("#90EE90", "100% correct (all models)"),
            ("#98FB98", "80-99% correct"),
            ("#ADFF2F", "60-79% correct"),
            ("#FFFF99", "40-59% correct (or all N/A)"),
            ("#FFB366", "20-39% correct"),
            ("#FF9999", "1-19% correct"),
            ("#FFB6C1", "0% correct (no models)")
        ]
        
        for color, description in legend_items:
            item_frame = ttk.Frame(legend_controls)
            item_frame.pack(side=tk.LEFT, padx=5)
            
            # Color indicator
            color_label = tk.Label(item_frame, text="■", fg=color, font=("Arial", 12))
            color_label.pack(side=tk.LEFT)
            
            # Description
            desc_label = ttk.Label(item_frame, text=description, font=("Arial", 9))
            desc_label.pack(side=tk.LEFT, padx=(2, 0))
        
        # Model accuracy display frame
        self.accuracy_frame = ttk.LabelFrame(main_frame, text="Model Accuracies (Filtered Results)")
        self.accuracy_frame.pack(fill=tk.X, pady=(10, 5))
        
        # Create accuracy display
        self.accuracy_display = ttk.Frame(self.accuracy_frame)
        self.accuracy_display.pack(fill=tk.X, padx=10, pady=5)
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT)
        
        # Load initial data
        self._refresh_view()
    
    def _refresh_view(self):
        """Refresh the results tree view based on filters with models as columns"""
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Get filter values
        selected_split = self.split_var.get()
        selected_group = self.group_var.get()
        selected_correct = self.correct_var.get()
        
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
        
        # Configure tags for different accuracy levels
        self.results_tree.tag_configure("perfect", background="#90EE90")  # 100% correct
        self.results_tree.tag_configure("excellent", background="#98FB98")  # 80-99% correct
        self.results_tree.tag_configure("good", background="#ADFF2F")  # 60-79% correct
        self.results_tree.tag_configure("neutral", background="#FFFF99")  # 40-59% correct or all N/A
        self.results_tree.tag_configure("poor", background="#FFB366")  # 20-39% correct
        self.results_tree.tag_configure("bad", background="#FF9999")  # 1-19% correct
        self.results_tree.tag_configure("terrible", background="#FFB6C1")  # 0% correct
        
        # Insert data into tree with enhanced visual indicators and color coding
        for data in combined_data:
            # Calculate accuracy for this row (how many models got it right)
            correct_count = 0
            total_models = len(model_names)
            na_count = 0
            
            # Count correct predictions (skip first 3 columns: tree_id, actual_group, split)
            for i in range(3, len(data)):
                cell_value = data[i]
                if isinstance(cell_value, str) and cell_value.startswith("✅"):
                    correct_count += 1
                elif isinstance(cell_value, str) and (cell_value == "N/A" or cell_value == "➖ N/A"):
                    na_count += 1
            
            # Adjust total models by removing N/A predictions
            total_models -= na_count
            
            # Calculate accuracy percentage
            if total_models > 0:
                accuracy_percentage = correct_count / total_models
            elif na_count == len(model_names):
                # All predictions are N/A (e.g., training data) - treat as neutral
                accuracy_percentage = 0.5  # Neutral color (yellow)
            else:
                accuracy_percentage = 0
            
            # Determine tag based on accuracy
            if accuracy_percentage == 1.0:
                tag = "perfect"
            elif accuracy_percentage >= 0.8:
                tag = "excellent"
            elif accuracy_percentage >= 0.6:
                tag = "good"
            elif accuracy_percentage >= 0.4:
                tag = "neutral"
            elif accuracy_percentage >= 0.2:
                tag = "poor"
            elif accuracy_percentage > 0:
                tag = "bad"
            else:
                tag = "terrible"
            
            # Insert row with tag
            item = self.results_tree.insert("", tk.END, values=data, tags=(tag,))
        
        # Update status with color coding information
        total_samples = len(combined_data)
        self.status_label.config(text=f"Showing {total_samples} samples across {len(model_names)} models | Row colors indicate how many models correctly classified each instance")
        
        # Calculate and display per-model accuracies
        self._update_model_accuracies(combined_data, model_names)
    
    def _update_model_accuracies(self, combined_data, model_names):
        """Update the model accuracy display based on filtered data"""
        # Clear existing accuracy labels
        for widget in self.accuracy_display.winfo_children():
            widget.destroy()
        
        # Calculate accuracies for each model
        model_accuracies = {}
        total_samples = len(combined_data)
        
        if total_samples == 0:
            ttk.Label(self.accuracy_display, text="No samples match the current filters", 
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
            model_frame = ttk.Frame(self.accuracy_display)
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
            self.accuracy_display.columnconfigure(i, weight=1)


class PredictionsViewer:
    """Handles the predictions viewer for 'Predict All' functionality"""
    
    def __init__(self, parent_window, all_predictions_df: pd.DataFrame):
        self.parent_window = parent_window
        self.all_predictions_df = all_predictions_df
        self.all_predictions = {}
        self.status_label = None
        self._convert_data()
        self._build_ui()
    
    def _convert_data(self):
        """Convert predictions dataframe to the format expected by ResultsViewer"""
        model_names = self.all_predictions_df['model'].unique()
        
        for model_name in model_names:
            model_data = self.all_predictions_df[self.all_predictions_df['model'] == model_name]
            
            # Create dataframe in the same format as main results viewer
            model_df = pd.DataFrame({
                'tree_id': model_data['instance_id'],
                'actual_group': model_data['actual'],
                'split': ['predict'] * len(model_data),  # All are predictions
                'predicted_group': model_data['prediction'],
                'is_correct': model_data['is_correct']
            })
            
            self.all_predictions[model_name] = model_df
    
    def _build_ui(self):
        """Build the predictions viewer UI (identical to ResultsViewer)"""
        # Main frame
        main_frame = ttk.Frame(self.parent_window)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Title
        title_label = ttk.Label(main_frame, text="Predictions on All Dataset Instances - Model Comparison", 
                               font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Filter controls frame
        filter_frame = ttk.LabelFrame(main_frame, text="Filters")
        filter_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Filter controls
        filter_controls = ttk.Frame(filter_frame)
        filter_controls.pack(fill=tk.X, padx=10, pady=10)
        
        # Split filter (not applicable for predictions, but keeping for consistency)
        ttk.Label(filter_controls, text="Split:").pack(side=tk.LEFT)
        self.split_var = tk.StringVar(value="All")
        split_cb = ttk.Combobox(filter_controls, textvariable=self.split_var, state="readonly", width=15)
        split_cb['values'] = ["All", "predict"]
        split_cb.pack(side=tk.LEFT, padx=(5, 20))
        
        # Group filter
        ttk.Label(filter_controls, text="Group:").pack(side=tk.LEFT)
        self.group_var = tk.StringVar(value="All Groups")
        group_cb = ttk.Combobox(filter_controls, textvariable=self.group_var, state="readonly", width=15)
        
        # Get unique groups from all predictions
        all_groups = set()
        for df in self.all_predictions.values():
            all_groups.update(df['actual_group'].unique())
        group_cb['values'] = ["All Groups"] + sorted([str(g) for g in all_groups])
        group_cb.pack(side=tk.LEFT, padx=(5, 20))
        
        # Correctness filter
        ttk.Label(filter_controls, text="Correctness:").pack(side=tk.LEFT)
        self.correct_var = tk.StringVar(value="All")
        correct_cb = ttk.Combobox(filter_controls, textvariable=self.correct_var, state="readonly", width=15)
        correct_cb['values'] = ["All", "Correct", "Incorrect"]
        correct_cb.pack(side=tk.LEFT, padx=(5, 20))
        
        # Refresh button
        refresh_btn = ttk.Button(filter_controls, text="Refresh", 
                                command=self._refresh_view)
        refresh_btn.pack(side=tk.RIGHT)
        
        # Results tree frame
        tree_frame = ttk.Frame(main_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create treeview for results with models as columns
        model_names = list(self.all_predictions.keys())
        columns = ["tree_id", "actual_group", "split"] + model_names
        self.results_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=20)
        
        # Configure columns
        self.results_tree.heading("tree_id", text="ID")
        self.results_tree.heading("actual_group", text="Actual")
        self.results_tree.heading("split", text="Split")
        
        # Configure model columns
        for model_name in model_names:
            self.results_tree.heading(model_name, text=model_name)
        
        # Set column widths
        self.results_tree.column("tree_id", width=60, anchor=tk.CENTER)
        self.results_tree.column("actual_group", width=80, anchor=tk.CENTER)
        self.results_tree.column("split", width=60, anchor=tk.CENTER)
        
        # Set model column widths
        for model_name in model_names:
            self.results_tree.column(model_name, width=100, anchor=tk.CENTER)
        
        # Add scrollbars
        tree_scroll_v = ttk.Scrollbar(tree_frame, orient="vertical", command=self.results_tree.yview)
        tree_scroll_h = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.results_tree.xview)
        self.results_tree.configure(yscrollcommand=tree_scroll_v.set, xscrollcommand=tree_scroll_h.set)
        
        # Pack treeview and scrollbars
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll_v.pack(side=tk.RIGHT, fill=tk.Y)
        tree_scroll_h.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Color legend frame
        legend_frame = ttk.LabelFrame(main_frame, text="Color Legend")
        legend_frame.pack(fill=tk.X, pady=(5, 10))
        
        # Create legend with color indicators
        legend_controls = ttk.Frame(legend_frame)
        legend_controls.pack(fill=tk.X, padx=10, pady=5)
        
        # Legend items
        legend_items = [
            ("#90EE90", "100% correct (all models)"),
            ("#98FB98", "80-99% correct"),
            ("#ADFF2F", "60-79% correct"),
            ("#FFFF99", "40-59% correct (or all N/A)"),
            ("#FFB366", "20-39% correct"),
            ("#FF9999", "1-19% correct"),
            ("#FFB6C1", "0% correct (no models)")
        ]
        
        for color, description in legend_items:
            item_frame = ttk.Frame(legend_controls)
            item_frame.pack(side=tk.LEFT, padx=5)
            
            # Color indicator
            color_label = tk.Label(item_frame, text="■", fg=color, font=("Arial", 12))
            color_label.pack(side=tk.LEFT)
            
            # Description
            desc_label = ttk.Label(item_frame, text=description, font=("Arial", 9))
            desc_label.pack(side=tk.LEFT, padx=(2, 0))
        
        # Model accuracy display frame
        self.accuracy_frame = ttk.LabelFrame(main_frame, text="Model Accuracies (Filtered Results)")
        self.accuracy_frame.pack(fill=tk.X, pady=(10, 5))
        
        # Create accuracy display
        self.accuracy_display = ttk.Frame(self.accuracy_frame)
        self.accuracy_display.pack(fill=tk.X, padx=10, pady=5)
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        self.status_label = ttk.Label(status_frame, text="Ready")
        self.status_label.pack(side=tk.LEFT)
        
        # Load initial data
        self._refresh_view()
    
    def _refresh_view(self):
        """Refresh the predictions results tree view based on filters"""
        # Use the same logic as ResultsViewer but with predictions data
        # Clear existing items
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Get filter values
        selected_split = self.split_var.get()
        selected_group = self.group_var.get()
        selected_correct = self.correct_var.get()
        
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
        
        # Configure tags for different accuracy levels
        self.results_tree.tag_configure("perfect", background="#90EE90")  # 100% correct
        self.results_tree.tag_configure("excellent", background="#98FB98")  # 80-99% correct
        self.results_tree.tag_configure("good", background="#ADFF2F")  # 60-79% correct
        self.results_tree.tag_configure("neutral", background="#FFFF99")  # 40-59% correct or all N/A
        self.results_tree.tag_configure("poor", background="#FFB366")  # 20-39% correct
        self.results_tree.tag_configure("bad", background="#FF9999")  # 1-19% correct
        self.results_tree.tag_configure("terrible", background="#FFB6C1")  # 0% correct
        
        # Insert data into tree with enhanced visual indicators and color coding
        for data in combined_data:
            # Calculate accuracy for this row (how many models got it right)
            correct_count = 0
            total_models = len(model_names)
            na_count = 0
            
            # Count correct predictions (skip first 3 columns: tree_id, actual_group, split)
            for i in range(3, len(data)):
                cell_value = data[i]
                if isinstance(cell_value, str) and cell_value.startswith("✅"):
                    correct_count += 1
                elif isinstance(cell_value, str) and (cell_value == "N/A" or cell_value == "➖ N/A"):
                    na_count += 1
            
            # Adjust total models by removing N/A predictions
            total_models -= na_count
            
            # Calculate accuracy percentage
            if total_models > 0:
                accuracy_percentage = correct_count / total_models
            elif na_count == len(model_names):
                # All predictions are N/A (e.g., training data) - treat as neutral
                accuracy_percentage = 0.5  # Neutral color (yellow)
            else:
                accuracy_percentage = 0
            
            # Determine tag based on accuracy
            if accuracy_percentage == 1.0:
                tag = "perfect"
            elif accuracy_percentage >= 0.8:
                tag = "excellent"
            elif accuracy_percentage >= 0.6:
                tag = "good"
            elif accuracy_percentage >= 0.4:
                tag = "neutral"
            elif accuracy_percentage >= 0.2:
                tag = "poor"
            elif accuracy_percentage > 0:
                tag = "bad"
            else:
                tag = "terrible"
            
            # Insert row with tag
            item = self.results_tree.insert("", tk.END, values=data, tags=(tag,))
        
        # Update status with color coding information
        total_samples = len(combined_data)
        self.status_label.config(text=f"Showing {total_samples} samples across {len(model_names)} models | Row colors indicate how many models correctly classified each instance")
        
        # Calculate and display per-model accuracies
        self._update_model_accuracies(combined_data, model_names)
    
    def _update_model_accuracies(self, combined_data, model_names):
        """Update the model accuracy display based on filtered data"""
        # Clear existing accuracy labels
        for widget in self.accuracy_display.winfo_children():
            widget.destroy()
        
        # Calculate accuracies for each model
        model_accuracies = {}
        for model_name in model_names:
            correct_count = 0
            total_count = 0
            
            # Find the column index for this model (skip first 3 columns)
            model_col_idx = 3 + model_names.index(model_name)
            
            for data in combined_data:
                cell_value = data[model_col_idx]
                if isinstance(cell_value, str) and not (cell_value == "N/A" or cell_value == "➖ N/A"):
                    total_count += 1
                    if cell_value.startswith("✅"):
                        correct_count += 1
            
            if total_count > 0:
                accuracy = correct_count / total_count
                model_accuracies[model_name] = accuracy
            else:
                model_accuracies[model_name] = 0.0
        
        # Sort models by accuracy (descending)
        sorted_models = sorted(model_accuracies.items(), key=lambda x: x[1], reverse=True)
        
        # Display accuracies in a single row
        for model_name, accuracy in sorted_models:
            # Create frame for this model's accuracy
            model_frame = ttk.Frame(self.accuracy_display)
            model_frame.pack(side=tk.LEFT, padx=10)
            
            # Model name
            name_label = ttk.Label(model_frame, text=f"{model_name}:", font=("Arial", 9, "bold"))
            name_label.pack(side=tk.LEFT)
            
            # Accuracy value with color coding
            accuracy_text = f"{accuracy:.3f} ({accuracy*100:.1f}%)"
            accuracy_label = ttk.Label(model_frame, text=accuracy_text, font=("Arial", 9))
            accuracy_label.pack(side=tk.LEFT, padx=(5, 0))
