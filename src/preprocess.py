from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class PreparedData:
	def __init__(self, X_train: np.ndarray, X_test: np.ndarray, y_train_enc: np.ndarray, y_test_enc: np.ndarray,
				 id_train: pd.Series, id_test: pd.Series, label_encoder: LabelEncoder, scaler: StandardScaler):
		self.X_train = X_train
		self.X_test = X_test
		self.y_train_enc = y_train_enc
		self.y_test_enc = y_test_enc
		self.id_train = id_train
		self.id_test = id_test
		self.label_encoder = label_encoder
		self.scaler = scaler


def train_test_prepare(df: pd.DataFrame, target: str, features: list[str], test_size: float = 0.3,
					  random_state: int = 42, stratify: bool = True,
					  id_column: str | None = None) -> PreparedData:
	# Separate features and target
	y = df[target]
	X = df[features]

	# Preserve id if provided
	id_train = pd.Series(dtype=float)
	id_test = pd.Series(dtype=float)
	if id_column and id_column in df.columns:
		id_series = df[id_column]
		X = X.drop(columns=[id_column]) if id_column in X.columns else X
	else:
		id_series = pd.Series([None] * len(df))

	# Split
	X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
		X, y, id_series, test_size=test_size, random_state=random_state,
		stratify=y if stratify else None
	)

	# Impute missing numeric values with mean
	imputer = SimpleImputer(strategy="mean")
	X_train_imp = imputer.fit_transform(X_train)
	X_test_imp = imputer.transform(X_test)

	# Scale
	scaler = StandardScaler()
	X_train_scaled = scaler.fit_transform(X_train_imp)
	X_test_scaled = scaler.transform(X_test_imp)

	# Encode target
	le = LabelEncoder()
	y_train_enc = le.fit_transform(y_train)
	y_test_enc = le.transform(y_test)

	return PreparedData(X_train_scaled, X_test_scaled, y_train_enc, y_test_enc, id_train, id_test, le, scaler)
