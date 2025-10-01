"""
UI Components module for muML
Handles styling, widgets, and basic UI elements
"""

import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Any, Callable


class UIStyles:
    """Handles UI styling and theme configuration"""
    
    def __init__(self):
        self.colors = {
            'primary': '#0D4F6B',      # Very dark blue for maximum contrast
            'secondary': '#6B1A4A',    # Very dark purple for maximum contrast
            'success': '#B8650A',      # Very dark orange for maximum contrast
            'danger': '#8B1A0F',       # Very dark red for maximum contrast
            'warning': '#D4B85A',      # Medium yellow with dark text
            'info': '#2A7A73',         # Very dark teal for maximum contrast
            'light': '#F7F7F7',        # Light gray
            'dark': '#2C3E50',         # Dark blue-gray
            'white': '#FFFFFF',
            'black': '#000000'         # Pure black for maximum contrast
        }
    
    def configure_styles(self, style: ttk.Style):
        """Configure modern styling for the application"""
        # Plain button styles with light backgrounds and dark text
        style.configure('Primary.TButton',
                       background='lightblue',  # Light blue
                       foreground='black',  # Black text
                       font=('Arial', 10, 'bold'))
        
        style.configure('Success.TButton',
                       background='lightgreen',  # Light green
                       foreground='black',  # Black text
                       font=('Arial', 10, 'bold'))
        
        style.configure('Danger.TButton',
                       background='lightcoral',  # Light red
                       foreground='black',  # Black text
                       font=('Arial', 10, 'bold'))
        
        style.configure('Info.TButton',
                       background='lightcyan',  # Light cyan
                       foreground='black',  # Black text
                       font=('Arial', 10, 'bold'))
        
        style.configure('Warning.TButton',
                       background='lightyellow',  # Light yellow
                       foreground='black',  # Black text
                       font=('Arial', 10, 'bold'))
        
        style.configure('Secondary.TButton',
                       background='plum',  # Light purple
                       foreground='black',  # Black text
                       font=('Arial', 10, 'bold'))
        
        # Configure frame styles
        style.configure('Card.TFrame',
                       background=self.colors['white'],
                       relief='solid',
                       borderwidth=1)
        
        # Configure label frame styles
        style.configure('Modern.TLabelframe',
                       background=self.colors['light'],
                       foreground=self.colors['dark'],
                       font=('Arial', 11, 'bold'))
        
        style.configure('Modern.TLabelframe.Label',
                       background=self.colors['light'],
                       foreground=self.colors['primary'],
                       font=('Arial', 11, 'bold'))
        
        # Configure entry styles
        style.configure('Modern.TEntry',
                       fieldbackground=self.colors['white'],
                       borderwidth=2,
                       relief='solid')
        
        # Configure combobox styles
        style.configure('Modern.TCombobox',
                       fieldbackground=self.colors['white'],
                       borderwidth=2,
                       relief='solid')
        
        # Configure treeview styles
        style.configure('Modern.Treeview',
                       background=self.colors['white'],
                       foreground=self.colors['dark'],
                       fieldbackground=self.colors['white'],
                       font=('Arial', 9))
        
        style.configure('Modern.Treeview.Heading',
                       background=self.colors['primary'],
                       foreground=self.colors['white'],
                       font=('Arial', 10, 'bold'))
        
        # Configure text widget
        style.configure('Modern.Text',
                       background=self.colors['white'],
                       foreground=self.colors['dark'],
                       font=('Consolas', 9))
        
        # Light background hover effects
        style.map('Primary.TButton',
                  background=[('active', 'skyblue'), ('pressed', 'steelblue')],
                  foreground=[('active', 'black'), ('pressed', 'white')])
        style.map('Success.TButton',
                  background=[('active', 'limegreen'), ('pressed', 'green')],
                  foreground=[('active', 'black'), ('pressed', 'white')])
        style.map('Danger.TButton',
                  background=[('active', 'salmon'), ('pressed', 'red')],
                  foreground=[('active', 'black'), ('pressed', 'white')])
        style.map('Info.TButton',
                  background=[('active', 'aqua'), ('pressed', 'blue')],
                  foreground=[('active', 'black'), ('pressed', 'white')])
        style.map('Warning.TButton',
                  background=[('active', 'gold'), ('pressed', 'orange')],
                  foreground=[('active', 'black'), ('pressed', 'white')])
        style.map('Secondary.TButton',
                  background=[('active', 'violet'), ('pressed', 'purple')],
                  foreground=[('active', 'black'), ('pressed', 'white')])


class FeatureSelector:
    """Handles feature selection UI components"""
    
    def __init__(self, parent_frame, on_target_changed: Callable, on_id_changed: Callable, on_feature_importance: Callable = None):
        self.parent_frame = parent_frame
        self.on_target_changed = on_target_changed
        self.on_id_changed = on_id_changed
        self.on_feature_importance = on_feature_importance
        self.feature_vars: Dict[str, tk.BooleanVar] = {}
        self._build_ui()
    
    def _build_ui(self):
        """Build the feature selection UI"""
        # Feature control buttons
        feat_ctrl = ttk.Frame(self.parent_frame)
        feat_ctrl.pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(feat_ctrl, text="✅ Select All", command=self._select_all_features, 
                  style='Success.TButton').pack(side=tk.LEFT, padx=(0, 3))
        ttk.Button(feat_ctrl, text="❌ Select None", command=self._select_none_features, 
                  style='Danger.TButton').pack(side=tk.LEFT, padx=3)
        
        # SHAP feature importance button (if callback provided)
        if self.on_feature_importance:
            ttk.Button(feat_ctrl, text="📈 Feature Importance", 
                      command=self.on_feature_importance, style='Info.TButton').pack(side=tk.RIGHT)
        
        # Scrollable checkbox area
        self.feat_canvas = tk.Canvas(self.parent_frame, borderwidth=0, highlightthickness=0)
        scroll = ttk.Scrollbar(self.parent_frame, orient="vertical", command=self.feat_canvas.yview)
        self.feat_inner = ttk.Frame(self.feat_canvas)
        self.feat_inner.bind(
            "<Configure>", lambda e: self.feat_canvas.configure(scrollregion=self.feat_canvas.bbox("all"))
        )
        self.feat_canvas.create_window((0, 0), window=self.feat_inner, anchor="nw")
        self.feat_canvas.configure(yscrollcommand=scroll.set)
        self.feat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=4)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=4)
    
    def rebuild_feature_checks(self, cols: List[str], current_target: str = "", current_id: str = ""):
        """Rebuild feature checkboxes"""
        # Clear existing
        for child in list(self.feat_inner.children.values()):
            child.destroy()
        self.feature_vars.clear()
        
        # Create a checkbutton for each column (skip target and id)
        for c in cols:
            if (current_target and c == current_target) or (current_id and c == current_id):
                continue
            # Set "Common Name" to unchecked by default, all others checked
            default_value = False if c == "Common Name" else True
            var = tk.BooleanVar(value=default_value)
            self.feature_vars[c] = var
            cb = ttk.Checkbutton(self.feat_inner, text=c, variable=var)
            cb.pack(anchor=tk.W, padx=4, pady=2)
    
    def _select_all_features(self):
        """Select all features"""
        for var in self.feature_vars.values():
            var.set(True)
    
    def _select_none_features(self):
        """Deselect all features"""
        for var in self.feature_vars.values():
            var.set(False)
    
    def get_selected_features(self) -> List[str]:
        """Get list of selected features"""
        selected = []
        for name, var in self.feature_vars.items():
            if var.get():
                selected.append(name)
        return selected


class AlgorithmSelector:
    """Handles algorithm selection UI components"""
    
    def __init__(self, parent_frame, algorithms: Dict[str, Any], on_configure_click: Callable):
        self.parent_frame = parent_frame
        self.algorithms = algorithms
        self.on_configure_click = on_configure_click
        self.selected_algos: Dict[str, tk.BooleanVar] = {k: tk.BooleanVar(value=True) for k in algorithms.keys()}
        self.algo_checks: Dict[str, ttk.Checkbutton] = {}
        self.algo_buttons: Dict[str, ttk.Button] = {}
        self._build_ui()
    
    def _build_ui(self):
        """Build the algorithm selection UI"""
        for name, var in self.selected_algos.items():
            # Checkbox for selection
            cb = ttk.Checkbutton(self.parent_frame, text=name, variable=var)
            cb.pack(anchor=tk.W, padx=8, pady=2)
            self.algo_checks[name] = cb
            
            # Button for hyperparameter configuration
            btn = ttk.Button(self.parent_frame, text=f"Configure {name}", 
                           command=lambda n=name: self.on_configure_click(n))
            btn.pack(anchor=tk.W, padx=8, pady=1)
            self.algo_buttons[name] = btn
    
    def get_selected_algorithms(self) -> List[str]:
        """Get list of selected algorithms"""
        return [name for name, var in self.selected_algos.items() if var.get()]
    
    def update_algorithms(self, new_algorithms: Dict[str, Any]):
        """Update algorithms and rebuild UI"""
        self.algorithms = new_algorithms
        # Clear existing
        for child in list(self.parent_frame.children.values()):
            child.destroy()
        self.algo_checks.clear()
        self.algo_buttons.clear()
        
        # Rebuild with new algorithms
        self.selected_algos = {k: tk.BooleanVar(value=True) for k in new_algorithms.keys()}
        self._build_ui()


class HyperparameterConfig:
    """Handles hyperparameter configuration UI"""
    
    def __init__(self, parent_frame):
        self.parent_frame = parent_frame
        self.hp_widgets: Dict[str, Dict[str, Any]] = {}
        self.search_strategies: Dict[str, tk.StringVar] = {}
        self.custom_params: Dict[str, Dict[str, Any]] = {}
        self._init_hyperparameter_widgets()
    
    def _init_hyperparameter_widgets(self):
        """Initialize hyperparameter widgets for all algorithms"""
        # This will be populated when algorithms are set
        pass
    
    def set_algorithms(self, algorithms: Dict[str, Any]):
        """Set algorithms and initialize their hyperparameter widgets"""
        for algo_name in algorithms.keys():
            self.hp_widgets[algo_name] = {}
            self.search_strategies[algo_name] = tk.StringVar(value="Grid Search")
            self.custom_params[algo_name] = {}
    
    def show_hyperparameters(self, algo_name: str, algorithms: Dict[str, Any]):
        """Show hyperparameter controls for selected algorithm"""
        if algo_name not in algorithms:
            return
        
        # Clear existing hyperparameter widgets
        for child in list(self.parent_frame.children.values()):
            child.destroy()
        
        # Clear the widget references for this algorithm to prevent stale references
        if algo_name in self.hp_widgets:
            del self.hp_widgets[algo_name]
        
        algo_info = algorithms[algo_name]
        params = algo_info.get("params", {})
        
        if not params:
            ttk.Label(self.parent_frame, text=f"No hyperparameters for {algo_name}", 
                     font=("Arial", 10, "italic")).pack(pady=10)
            return
        
        # Search strategy selection
        strategy_frame = ttk.Frame(self.parent_frame)
        strategy_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(strategy_frame, text="Search Strategy:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        ttk.Radiobutton(strategy_frame, text="Grid Search", variable=self.search_strategies[algo_name], 
                       value="Grid Search").pack(side=tk.LEFT, padx=(10, 5))
        ttk.Radiobutton(strategy_frame, text="Random Search", variable=self.search_strategies[algo_name], 
                       value="Random Search").pack(side=tk.LEFT, padx=5)
        
        # Hyperparameter controls
        ttk.Label(self.parent_frame, text=f"Hyperparameters for {algo_name}:", 
                 font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # Initialize fresh widget dictionary for this algorithm
        self.hp_widgets[algo_name] = {}
        
        for param_name, default_values in params.items():
            param_frame = ttk.Frame(self.parent_frame)
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
    
    def show_initial_message(self):
        """Show initial message in hyperparameters frame"""
        # Clear hyperparameter area
        for child in list(self.parent_frame.children.values()):
            child.destroy()
        
        # Clear all widget references to prevent stale references
        self.hp_widgets.clear()
        
        ttk.Label(self.parent_frame, text="Click 'Configure [Algorithm]' to set hyperparameters", 
                 font=("Arial", 10, "italic")).pack(pady=20)
        
        # Add instruction text
        instruction_text = """Instructions:
• Click "Configure [Algorithm]" to set hyperparameters
• Enter values as lists: [1, 2, 3] or single values: 5
• Choose Grid Search for exhaustive search or Random Search for sampling
• Leave empty to use default values
• Check algorithms to include them in training
• Adjust n_jobs to control CPU usage (8 = default, -1 = all cores, 1 = single core)"""
        ttk.Label(self.parent_frame, text=instruction_text, font=("Arial", 8), 
                 foreground="gray", justify=tk.LEFT).pack(pady=10, padx=10)
    
    def get_custom_hyperparameters(self, selected_algorithms: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get custom hyperparameters from UI"""
        import ast
        custom_params = {}
        
        for algo_name in selected_algorithms:
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
                        print(f"Warning: Invalid parameter value for {algo_name}.{param_name}: {e}")
                    continue
            
            if algo_params:
                custom_params[algo_name] = algo_params
        
        return custom_params


class ResultsTable:
    """Handles the results table display"""
    
    def __init__(self, parent_frame, on_table_click: Callable):
        self.parent_frame = parent_frame
        self.on_table_click = on_table_click
        self.results_table = None
        self._build_ui()
    
    def _build_ui(self):
        """Build the results table UI"""
        table_container = ttk.Frame(self.parent_frame)
        table_container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        self.results_table = ttk.Treeview(table_container, columns=("Classifier", "Accuracy", "F1-Score"), 
                                        show="headings", style='Modern.Treeview')
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
        self.results_table.bind("<Double-1>", self.on_table_click)
    
    def clear_results(self):
        """Clear all results from the table"""
        for row in self.results_table.get_children():
            self.results_table.delete(row)
    
    def show_results(self, results_df):
        """Display results in the table"""
        self.clear_results()
        for _, r in results_df.iterrows():
            sclf = str(r.get("Classifier"))
            acc = f"{float(r.get('Accuracy', 0.0)):.4f}"
            f1 = f"{float(r.get('F1-Score', 0.0)):.4f}"
            self.results_table.insert("", tk.END, values=(sclf, acc, f1))
