from __future__ import annotations

import io
from typing import Optional

import pandas as pd


def read_csv(path_or_buffer: str | io.BytesIO, encoding: Optional[str] = None) -> pd.DataFrame:
	return pd.read_csv(path_or_buffer, encoding=encoding)  # type: ignore[arg-type]


def list_columns(df: pd.DataFrame) -> list[str]:
	return list(df.columns)
