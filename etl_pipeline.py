#!/usr/bin/env python3
"""
Data Pipeline Development: automated ETL (Extract -> Transform -> Load)

Tools: pandas, NumPy, scikit-learn, SQLAlchemy (SQLite)

Usage:
    python etl_pipeline.py                                  # auto-generates sample data if missing
    python etl_pipeline.py --generate-sample --rows 1000    # force-regenerate sample data
    python etl_pipeline.py --input data/raw/data.csv        # run on your own CSV
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from sqlalchemy import create_engine
except ImportError:  # fallback to the standard-library driver
    create_engine = None
import sqlite3

BASE_DIR = Path(__file__).resolve().parent
LOG_FILE = BASE_DIR / "logs" / "pipeline.log"

# Schema used for validation and column typing (after name standardization)
ID_COL = "customer_id"
TARGET_COL = "churned"
DATE_COL = "last_purchase_date"
NUMERIC_COLS = ["age", "annual_income", "spending_score", "tenure_months", "days_since_last_purchase"]
CATEGORICAL_COLS = ["gender", "city", "subscription_type"]
REQUIRED_COLS = [ID_COL, TARGET_COL, DATE_COL, "age", "annual_income", "spending_score",
                 "tenure_months", "gender", "city", "subscription_type"]

logger = logging.getLogger("etl")


def setup_logging() -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler(sys.stdout)],
    )


# --------------------------------------------------------------------------- #
# Sample data generator
# --------------------------------------------------------------------------- #
def generate_sample_data(path: Path, rows: int = 1000, seed: int = 42) -> None:
    """Create a messy customer dataset: missing values, duplicates, outliers, inconsistent text."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "Customer ID": np.arange(1, rows + 1),
        "Age": rng.integers(18, 70, rows).astype(float),
        "Gender": rng.choice(["Male", "Female", "Other"], rows, p=[0.48, 0.48, 0.04]),
        "City": rng.choice(["Ludhiana", "Delhi", "Mumbai", "Pune", "Chandigarh", "Jaipur"], rows),
        "Annual Income": rng.normal(60000, 18000, rows).round(2),
        "Spending Score": rng.integers(1, 100, rows).astype(float),
        "Tenure Months": rng.integers(1, 120, rows).astype(float),
        "Subscription Type": rng.choice(["Basic", "Standard", "Premium"], rows, p=[0.5, 0.3, 0.2]),
        "Last Purchase Date": pd.to_datetime("2026-09-30") - pd.to_timedelta(rng.integers(0, 365, rows), unit="D"),
        "Churned": rng.choice([0, 1], rows, p=[0.75, 0.25]),
    })
    df["Last Purchase Date"] = df["Last Purchase Date"].dt.strftime("%Y-%m-%d")

    # Inject missing values (~5-8% per column)
    for col in ["Age", "Gender", "City", "Annual Income", "Spending Score", "Subscription Type", "Last Purchase Date"]:
        df.loc[rng.choice(rows, int(rows * rng.uniform(0.05, 0.08)), replace=False), col] = np.nan

    # Inject outliers and invalid values
    out_idx = rng.choice(rows, 12, replace=False)
    df.loc[out_idx[:6], "Annual Income"] = rng.uniform(400000, 900000, 6)
    df.loc[out_idx[6:9], "Age"] = [150, 999, -5]
    df.loc[out_idx[9:], "Tenure Months"] = [900, 1200, 2000]

    # Inconsistent casing / whitespace in text columns
    mask = df["City"].notna()
    noisy = rng.choice(mask[mask].index, 80, replace=False)
    df.loc[noisy, "City"] = df.loc[noisy, "City"].str.upper().radd("  ")

    # Duplicate rows
    df = pd.concat([df, df.sample(25, random_state=seed)], ignore_index=True)

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Sample dataset generated: %s (%d rows)", path, len(df))


# --------------------------------------------------------------------------- #
# Extract
# --------------------------------------------------------------------------- #
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (df.columns.str.strip().str.lower()
                  .str.replace(r"[^0-9a-z]+", "_", regex=True).str.strip("_"))
    return df


def extract(path: Path) -> pd.DataFrame:
    logger.info("EXTRACT: reading %s", path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = standardize_columns(pd.read_csv(path))
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Schema validation failed, missing columns: {missing}")
    logger.info("EXTRACT: %d rows x %d columns loaded", *df.shape)
    logger.info("EXTRACT: missing values per column: %s", df.isna().sum()[df.isna().sum() > 0].to_dict())
    return df


# --------------------------------------------------------------------------- #
# Transform
# --------------------------------------------------------------------------- #
class IQRClipper(BaseEstimator, TransformerMixin):
    """Clip each numeric column to [Q1 - k*IQR, Q3 + k*IQR]; NaN-safe, learns bounds on fit."""

    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, 25, axis=0), np.nanpercentile(X, 75, axis=0)
        iqr = q3 - q1
        self.lower_, self.upper_ = q1 - self.factor * iqr, q3 + self.factor * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level cleaning that is not a learned transformation."""
    before = len(df)
    df = df.drop_duplicates().copy()
    logger.info("TRANSFORM: dropped %d duplicate rows", before - len(df))

    # Text normalization
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(object).str.strip().str.title()

    # Type coercion
    for col in ["age", "annual_income", "spending_score", "tenure_months"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")

    # Invalid values become NaN so they get imputed (age must be 0-100, tenure non-negative)
    invalid_age = ~df["age"].between(0, 100) & df["age"].notna()
    df.loc[invalid_age, "age"] = np.nan
    df.loc[df["tenure_months"] < 0, "tenure_months"] = np.nan
    logger.info("TRANSFORM: %d invalid ages set to NaN for imputation", invalid_age.sum())

    # Feature engineering: recency from date
    ref_date = df[DATE_COL].max()
    df["days_since_last_purchase"] = (ref_date - df[DATE_COL]).dt.days

    # Drop rows with no ID or no target label (cannot be imputed meaningfully)
    df = df.dropna(subset=[ID_COL, TARGET_COL])
    return df.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline([
        ("clip", IQRClipper(factor=1.5)),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric_pipe, NUMERIC_COLS), ("cat", categorical_pipe, CATEGORICAL_COLS)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def transform(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("TRANSFORM: cleaning data")
    df = clean(df)

    logger.info("TRANSFORM: fitting preprocessing pipeline (clip -> impute -> scale / encode)")
    preprocessor = build_preprocessor()
    matrix = preprocessor.fit_transform(df)
    features = pd.DataFrame(matrix, columns=preprocessor.get_feature_names_out())

    out = pd.concat([df[[ID_COL]].astype(int), features, df[[TARGET_COL]].astype(int)], axis=1)
    assert out.drop(columns=[ID_COL, TARGET_COL]).isna().sum().sum() == 0, "NaNs remain after transform"
    logger.info("TRANSFORM: output shape %s, no missing values remaining", out.shape)
    return out


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
def load(df: pd.DataFrame, out_dir: Path, table: str = "customers_processed") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "customers_processed.csv"
    df.to_csv(csv_path, index=False)
    logger.info("LOAD: CSV written to %s", csv_path)

    db_path = out_dir / "pipeline.db"
    # SQLAlchemy if installed, otherwise the built-in sqlite3 driver
    conn = create_engine(f"sqlite:///{db_path}").connect() if create_engine else sqlite3.connect(db_path)
    try:
        df.to_sql(table, conn, if_exists="replace", index=False)
        count = pd.read_sql(f"SELECT COUNT(*) AS n FROM {table}", conn)["n"].iloc[0]
    finally:
        conn.close()
    logger.info("LOAD: %d rows written to SQLite table '%s' (%s)", count, table, db_path)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run(input_path: Path, output_dir: Path) -> None:
    raw = extract(input_path)
    processed = transform(raw)
    load(processed, output_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETL pipeline: pandas + scikit-learn + SQLite")
    p.add_argument("--input", type=Path, default=BASE_DIR / "data" / "raw" / "data.csv", help="input CSV path")
    p.add_argument("--output-dir", type=Path, default=BASE_DIR / "data" / "processed", help="output folder")
    p.add_argument("--generate-sample", action="store_true", help="(re)generate the sample dataset first")
    p.add_argument("--rows", type=int, default=1000, help="rows in the sample dataset")
    return p.parse_args()


def main() -> None:
    setup_logging()
    args = parse_args()
    try:
        if args.generate_sample or not args.input.exists():
            generate_sample_data(args.input, rows=args.rows)
        logger.info("=== ETL run started ===")
        run(args.input, args.output_dir)
        logger.info("=== ETL run finished successfully ===")
    except Exception:
        logger.exception("ETL run failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
