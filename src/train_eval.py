from __future__ import annotations

from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, cohen_kappa_score, confusion_matrix

from src.models import get_model_registry, build_search
from src.preprocess import PreparedData


def calculate_detailed_accuracy_metrics(y_true, y_pred):
	"""Calculate detailed accuracy metrics including overall, producer's, mean class, and kappa accuracy"""
	
	# Overall accuracy (same as accuracy_score)
	overall_accuracy = accuracy_score(y_true, y_pred)
	
	# Kappa accuracy (Cohen's Kappa)
	kappa_accuracy = cohen_kappa_score(y_true, y_pred)
	
	# Confusion matrix for producer's accuracy and mean class accuracy
	cm = confusion_matrix(y_true, y_pred)
	
	# Producer's accuracy (recall for each class) - diagonal / row sums
	producer_accuracies = []
	class_labels = sorted(list(set(y_true) | set(y_pred)))
	
	for i, class_label in enumerate(class_labels):
		if i < cm.shape[0]:  # Make sure we don't go out of bounds
			row_sum = cm[i, :].sum()
			if row_sum > 0:
				producer_acc = cm[i, i] / row_sum
				producer_accuracies.append(producer_acc)
			else:
				producer_accuracies.append(0.0)
		else:
			producer_accuracies.append(0.0)
	
	# Mean class accuracy (average of producer's accuracies)
	mean_class_accuracy = np.mean(producer_accuracies) if producer_accuracies else 0.0
	
	return {
		"overall_accuracy": overall_accuracy,
		"producer_accuracy": producer_accuracies,
		"mean_class_accuracy": mean_class_accuracy,
		"kappa_accuracy": kappa_accuracy,
		"class_labels": class_labels
	}


def train_selected(models: List[str], prep: PreparedData, n_jobs: int = -1, random_state: int = 42, 
				  progress_callback=None, custom_hyperparams: Dict[str, Dict[str, Any]] = None, 
				  search_strategies: Dict[str, str] = None, stop_check_callback=None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
	registry = get_model_registry(random_state)
	results: List[Dict[str, Any]] = []
	detailed_results: Dict[str, Dict[str, Any]] = {}
	best_key = None
	best_f1 = -1.0
	best_estimator = None
	
	# Determine appropriate CV folds based on smallest class size
	from collections import Counter
	class_counts = Counter(prep.y_train_enc)
	min_class_size = min(class_counts.values())
	# Use minimum of 5 folds or smallest class size, but at least 2
	cv_folds = min(5, max(2, min_class_size))
	
	if progress_callback:
		progress_callback(f"Using {cv_folds}-fold CV (smallest class has {min_class_size} samples)")

	for i, name in enumerate(models):
		# Check if we should stop training
		if stop_check_callback and stop_check_callback():
			if progress_callback:
				progress_callback("Training stopped by user.")
			break
			
		if name not in registry:
			continue
		
		if progress_callback:
			progress_callback(f"Training {name} ({i+1}/{len(models)})...")
		
		import time
		start_time = time.time()
		
		info = registry[name]
		
		# Get custom hyperparameters and search strategy for this algorithm
		custom_params = custom_hyperparams.get(name, {}) if custom_hyperparams else {}
		search_strategy = search_strategies.get(name, "Grid Search") if search_strategies else "Grid Search"
		
		search = build_search(info, scoring="f1_weighted", cv=cv_folds, n_jobs=n_jobs, random_state=random_state,
							custom_params=custom_params, search_strategy=search_strategy)
		search.fit(prep.X_train, prep.y_train_enc)
		best_model = search.best_estimator_

		y_pred_enc = best_model.predict(prep.X_test)
		y_pred = prep.label_encoder.inverse_transform(y_pred_enc)
		y_test = prep.label_encoder.inverse_transform(prep.y_test_enc)

		acc = accuracy_score(y_test, y_pred)
		f1 = f1_score(y_test, y_pred, average="weighted")
		crep = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
		
		# Calculate detailed accuracy metrics
		detailed_metrics = calculate_detailed_accuracy_metrics(y_test, y_pred)
		
		elapsed = time.time() - start_time

		row = {
			"Classifier": name,
			"Accuracy": acc,
			"Mean Class Acc": detailed_metrics["mean_class_accuracy"],
			"Kappa Acc": detailed_metrics["kappa_accuracy"],
			"F1-Score": f1,
		}
		
		# Store detailed results for popup
		detailed_results[name] = {
			"accuracy": acc,
			"f1_weighted": f1,
			"classification_report": crep,
			"training_time": elapsed,
			"y_test": y_test,
			"y_pred": y_pred,
			"best_params": search.best_params_,
			"cv_results": search.cv_results_,
			"best_estimator": best_model,
			"detailed_accuracy_metrics": detailed_metrics
		}
		
		# add per-class for up to 5 classes if present
		for cls in ["1.0", "2.0", "3.0", "4.0", "5.0"]:
			if cls in crep:
				row[f"Class {cls} Precision"] = crep[cls]["precision"]
				row[f"Class {cls} Recall"] = crep[cls]["recall"]
				row[f"Class {cls} F1"] = crep[cls]["f1-score"]

		results.append(row)
		
		if progress_callback:
			progress_callback(f"✓ {name} completed in {elapsed:.1f}s - Accuracy: {acc:.3f}, F1: {f1:.3f}")

		if f1 > best_f1:
			best_f1 = f1
			best_key = name
			best_estimator = best_model

	results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False).reset_index(drop=True)
	return results_df, {"best_name": best_key, "best_estimator": best_estimator, "detailed_results": detailed_results}


def build_predictions_df(prep: PreparedData, best_estimator) -> pd.DataFrame:
	# Predict test set
	y_pred_test_enc = best_estimator.predict(prep.X_test)
	y_pred_test = prep.label_encoder.inverse_transform(y_pred_test_enc)
	y_test = prep.label_encoder.inverse_transform(prep.y_test_enc)

	test_df = pd.DataFrame({
		"tree_id": prep.id_test,
		"actual_group": y_test,
		"split": "test",
		"predicted_group": y_pred_test,
	})
	test_df["is_correct"] = test_df["actual_group"] == test_df["predicted_group"]

	# Train rows as NA for predictions
	train_df = pd.DataFrame({
		"tree_id": prep.id_train,
		"actual_group": prep.label_encoder.inverse_transform(prep.y_train_enc),
		"split": "train",
		"predicted_group": np.nan,
		"is_correct": np.nan,
	})

	final = pd.concat([train_df, test_df], ignore_index=True)
	final = final.sort_values(by="tree_id").reset_index(drop=True)
	return final


def build_all_predictions_df(prep: PreparedData, detailed_results: Dict[str, Dict[str, Any]]) -> Dict[str, pd.DataFrame]:
	"""Build prediction dataframes for all trained models"""
	all_predictions = {}
	
	for model_name, results in detailed_results.items():
		best_estimator = results.get("best_estimator")
		if best_estimator is None:
			continue
		
		# Predict test set
		y_pred_test_enc = best_estimator.predict(prep.X_test)
		y_pred_test = prep.label_encoder.inverse_transform(y_pred_test_enc)
		y_test = prep.label_encoder.inverse_transform(prep.y_test_enc)

		test_df = pd.DataFrame({
			"tree_id": prep.id_test,
			"actual_group": y_test,
			"split": "test",
			"predicted_group": y_pred_test,
		})
		test_df["is_correct"] = test_df["actual_group"] == test_df["predicted_group"]

		# Train rows with actual labels (no predictions for training data)
		train_df = pd.DataFrame({
			"tree_id": prep.id_train,
			"actual_group": prep.label_encoder.inverse_transform(prep.y_train_enc),
			"split": "train",
			"predicted_group": np.nan,
			"is_correct": np.nan,
		})

		final = pd.concat([train_df, test_df], ignore_index=True)
		final = final.sort_values(by="tree_id").reset_index(drop=True)
		all_predictions[model_name] = final
	
	return all_predictions
