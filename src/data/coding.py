from __future__ import annotations

import re
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


_NORMALIZE_RE = re.compile(r"[^0-9a-zA-Z]+")


def _normalize_name(name: str) -> str:
    return _NORMALIZE_RE.sub("_", name).strip("_").lower()


def normalize_column_names(df: pd.DataFrame) -> Dict[str, str]:
    """Return a normalized-name -> exact-name mapping for df columns.

    This does not modify the DataFrame. It exists to support resilient lookups
    across files where column casing/spacing might differ.
    """

    mapping: Dict[str, str] = {}
    collisions: Dict[str, list[str]] = {}

    for col in df.columns.astype(str).tolist():
        norm = _normalize_name(col)
        if norm in mapping and mapping[norm] != col:
            collisions.setdefault(norm, sorted({mapping[norm], col}))
        mapping[norm] = col

    if collisions:
        raise ValueError(f"Normalized column name collisions: {collisions}")

    return mapping


def recode_binary_yn(
    series: pd.Series,
    *,
    yes_values: Tuple = (1,),
    no_values: Tuple = (2,),
    missing_values: Tuple = (7, 9, 97, 99),
) -> pd.Series:
    """Recode YRBS-style binary codes to {0,1,NA}.

    - yes_values map to 1
    - no_values map to 0
    - missing_values map to NA
    - NaN stays NA

    Raises ValueError if unexpected non-missing values are observed.
    """

    s = series
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")

    is_yes = s.isin(yes_values)
    is_no = s.isin(no_values)
    is_missing_code = s.isin(missing_values)
    is_na = s.isna()

    out.loc[is_yes] = 1
    out.loc[is_no] = 0
    out.loc[is_missing_code | is_na] = pd.NA

    unexpected = s.loc[~(is_yes | is_no | is_missing_code | is_na)].dropna().unique()
    if len(unexpected) > 0:
        raise ValueError(
            "Unexpected codes in binary variable. "
            f"Observed unexpected values: {sorted(map(str, unexpected))}; "
            f"expected yes={yes_values}, no={no_values}, missing={missing_values}."
        )

    return out


def coerce_categorical(series: pd.Series, *, numeric_to_string: bool = True, prefix: str = "cat_") -> pd.Series:
    """Convert a Series to pandas categorical dtype.

    If numeric_to_string=True and the series is numeric, non-missing values are
    mapped to strings like 'cat_1' to avoid accidental ordinality.
    """

    s = series.copy()

    if numeric_to_string and pd.api.types.is_numeric_dtype(s):

        def _to_cat(v):
            if pd.isna(v):
                return np.nan
            fv = float(v)
            if fv.is_integer():
                return f"{prefix}{int(fv)}"
            return f"{prefix}{fv}"

        s = s.map(_to_cat)
    else:
        # Preserve missingness; normalize to pandas string dtype for consistency.
        s = s.astype("string")

    return s.astype("category")


def summarize_missingness(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-column missingness summary in stable column order."""

    n = len(df)
    rows = []
    for col in df.columns.astype(str).tolist():
        n_missing = int(df[col].isna().sum())
        missing_rate = round(n_missing / n, 6) if n else np.nan
        rows.append({"column": col, "n": n, "n_missing": n_missing, "missing_rate": missing_rate})
    return pd.DataFrame(rows)

