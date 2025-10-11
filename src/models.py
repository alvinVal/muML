from __future__ import annotations

from typing import Dict, Any

from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

try:
	from xgboost import XGBClassifier
	has_xgb = True
except Exception:
	has_xgb = False
	XGBClassifier = None  # type: ignore

SearchSpec = Dict[str, Any]

def get_model_registry(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
	registry: Dict[str, Dict[str, Any]] = {
		"AdaBoost": {
			"model": AdaBoostClassifier(random_state=random_state),
			"params": {
				"n_estimators": [10, 50, 75, 100, 125, 150, 175, 200],
				"learning_rate": [0.01, 0.1, 0.5, 1.0, 2.0],
			},
			"search": GridSearchCV,
		},
		"Decision Tree": {
			"model": DecisionTreeClassifier(random_state=random_state, class_weight="balanced"),
			"params": {
				"max_depth": [None, 10, 20, 30],
				"min_samples_split": [2, 5, 10],
				"min_samples_leaf": [1, 2, 4],
			},
			"search": GridSearchCV,
		},
		"Random Forest": {
			"model": RandomForestClassifier(random_state=random_state, class_weight="balanced"),
			"params": {
				"n_estimators": [100, 150, 200, 250, 300, 350, 400, 500],
				"max_depth": [None, 10, 20, 30, 40, 50],
				"min_samples_split": [2, 5, 10, 15, 20],
				"max_features": ["sqrt", "log2"],
			},
			"search": GridSearchCV,
		},
		"KNN": {
			"model": KNeighborsClassifier(),
			"params": {
				"n_neighbors": [1, 2, 3, 4, 5, 6, 7],
				"weights": ["uniform", "distance"],
			},
			"search": GridSearchCV,
		},
		"Naive Bayes": {
			"model": GaussianNB(),
			"params": {},
			"search": GridSearchCV,
		},
		"SVM": {
			"model": SVC(random_state=random_state, class_weight="balanced", probability=True),
			"params": {
				"C": [0.1, 1, 10, 15, 20],
				"kernel": ["rbf", "linear"],
			},
			"search": GridSearchCV,
		},
		"Logistic Regression": {
			"model": LogisticRegression(random_state=random_state, class_weight="balanced", max_iter=5000, solver="lbfgs"),
			"params": {
				"C": [0.01, 0.1, 1, 10, 100],
				"solver": ["lbfgs"],
			},
			"search": GridSearchCV,
		},
	}
	if has_xgb and XGBClassifier is not None:
		registry["XGBoost"] = {
			"model": XGBClassifier(random_state=random_state, eval_metric="mlogloss"),
            "params": {
                "max_depth": [9, 12, 15],
                "learning_rate": [0.3, 0.4, 0.5],
                "n_estimators": [125, 150, 175],
                "subsample": [0.7, 0.8, 0.9, 1.0],
                "colsample_bytree": [0.5, 0.75],
            },
			"search": GridSearchCV,
		}
	return registry


def build_search(model_info: Dict[str, Any], scoring: str = "f1_weighted", cv: int = 5, n_jobs: int = -1, 
				random_state: int = 42, custom_params: Dict[str, Any] = None, search_strategy: str = "Grid Search") -> Any:
	SearchClass = model_info["search"]
	base: SearchSpec = {
		"estimator": model_info["model"],
		"cv": cv,
		"scoring": scoring,
		"n_jobs": n_jobs,
	}
	
	# Use custom parameters if provided, otherwise use defaults
	params = custom_params if custom_params else model_info.get("params", {})
	
	# Override search strategy if specified
	if search_strategy == "Random Search":
		SearchClass = RandomizedSearchCV
		base["param_distributions"] = params
		base["n_iter"] = 20
		base["random_state"] = random_state
	elif search_strategy == "Grid Search":
		SearchClass = GridSearchCV
		base["param_grid"] = params
	else:
		# Use original search strategy
		if SearchClass == RandomizedSearchCV:
			base["param_distributions"] = params
			base["n_iter"] = 20
			base["random_state"] = random_state
		else:
			base["param_grid"] = params
	
	return SearchClass(**base)
