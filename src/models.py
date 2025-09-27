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


def get_model_registry() -> Dict[str, Dict[str, Any]]:
	registry: Dict[str, Dict[str, Any]] = {
		"AdaBoost": {
			"model": AdaBoostClassifier(random_state=42),
			"params": {
				"n_estimators": [50, 100, 200],
				"learning_rate": [0.01, 0.1, 1.0],
			},
			"search": GridSearchCV,
		},
		"Decision Tree": {
			"model": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
			"params": {
				"max_depth": [None, 10, 20, 30],
				"min_samples_split": [2, 5, 10],
				"min_samples_leaf": [1, 2, 4],
			},
			"search": RandomizedSearchCV,
		},
		"Random Forest": {
			"model": RandomForestClassifier(random_state=42, class_weight="balanced"),
			"params": {
				"n_estimators": [100, 200, 300],
				"max_depth": [None, 10, 20],
				"min_samples_split": [2, 5, 10],
				"max_features": ["sqrt", "log2"],
			},
			"search": RandomizedSearchCV,
		},
		"KNN": {
			"model": KNeighborsClassifier(),
			"params": {
				"n_neighbors": [3, 5, 7, 9],
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
			"model": SVC(random_state=42, class_weight="balanced", probability=True),
			"params": {
				"C": [0.1, 1, 10],
				"kernel": ["rbf", "linear"],
			},
			"search": GridSearchCV,
		},
		"Logistic Regression": {
			"model": LogisticRegression(random_state=42, class_weight="balanced", max_iter=5000, solver="lbfgs"),
			"params": {
				"C": [0.01, 0.1, 1, 10, 100],
				"solver": ["lbfgs"],
			},
			"search": GridSearchCV,
		},
	}
	if has_xgb and XGBClassifier is not None:
		registry["XGBoost"] = {
			"model": XGBClassifier(random_state=42, eval_metric="mlogloss"),
			"params": {
				"max_depth": [3, 5, 7],
				"learning_rate": [0.01, 0.1, 0.2],
				"n_estimators": [100, 200, 400],
				"subsample": [0.5, 0.8, 1.0],
				"colsample_bytree": [0.5, 0.75, 1.0],
			},
			"search": RandomizedSearchCV,
		}
	return registry


def build_search(model_info: Dict[str, Any], scoring: str = "f1_weighted", cv: int = 5, n_jobs: int = -1) -> Any:
	SearchClass = model_info["search"]
	base: SearchSpec = {
		"estimator": model_info["model"],
		"cv": cv,
		"scoring": scoring,
		"n_jobs": n_jobs,
	}
	params = model_info.get("params", {})
	if SearchClass == RandomizedSearchCV:
		base["param_distributions"] = params
		base["n_iter"] = 20
		base["random_state"] = 42
	else:
		base["param_grid"] = params
	return SearchClass(**base)
