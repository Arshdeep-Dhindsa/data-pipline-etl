# Data Pipeline Development: ETL with pandas and scikit-learn

An automated ETL (Extract, Transform, Load) pipeline that cleans a raw customer dataset, preprocesses it with scikit-learn, and loads the result into SQLite and CSV.

## Tech stack

- Python 3.10+
- pandas, NumPy: extraction and cleaning
- scikit-learn: `Pipeline` and `ColumnTransformer` for imputation, outlier clipping, scaling, encoding
- SQLite (SQLAlchemy if installed, built-in `sqlite3` otherwise): load target
- logging, argparse: run logs and command-line options

## Project structure

```
data_pipeline/
├── etl_pipeline.py        # the full ETL script
├── requirements.txt
├── README.md
├── data/
│   ├── raw/data.csv       # input (auto-generated sample if missing)
│   └── processed/         # customers_processed.csv + pipeline.db
└── logs/pipeline.log      # created on first run
```

## Setup

```bash
python -m venv venv
# Windows:      venv\Scripts\activate
# macOS/Linux:  source venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python etl_pipeline.py                                  # run on data/raw/data.csv (sample data is generated if missing)
python etl_pipeline.py --generate-sample --rows 5000    # regenerate a larger sample dataset
python etl_pipeline.py --input path/to/your.csv         # run on your own CSV
python etl_pipeline.py --output-dir out/                # change the output folder
```

## Pipeline stages

### 1. Extract
- Reads the CSV and standardizes column names (`Annual Income` becomes `annual_income`).
- Validates that all required columns exist.
- Logs row counts and missing values per column.

### 2. Transform
- Drops duplicate rows.
- Strips whitespace and normalizes casing in text columns.
- Coerces numeric and date types.
- Marks invalid values (age outside 0-100, negative tenure) as missing.
- Adds the feature `days_since_last_purchase` from the date column.
- Applies a scikit-learn `ColumnTransformer`:
  - **Numeric:** IQR outlier clipping, then median imputation, then `StandardScaler`.
  - **Categorical:** most-frequent imputation, then `OneHotEncoder`.
- Checks that no missing values remain.

### 3. Load
- Writes `data/processed/customers_processed.csv`.
- Writes the SQLite table `customers_processed` in `data/processed/pipeline.db`.

## Sample dataset

Generated automatically with realistic problems for the pipeline to fix: roughly 5-8% missing values per column, 25 duplicate rows, extreme income, age and tenure outliers, and inconsistent city casing and whitespace.

Columns: `customer_id, age, gender, city, annual_income, spending_score, tenure_months, subscription_type, last_purchase_date, churned`

## Using your own data

Your CSV needs the same columns as the sample. To use different columns, edit the constants at the top of `etl_pipeline.py`: `ID_COL`, `TARGET_COL`, `DATE_COL`, `NUMERIC_COLS`, `CATEGORICAL_COLS`, `REQUIRED_COLS`.

## Inspecting the output

```bash
sqlite3 data/processed/pipeline.db "SELECT * FROM customers_processed LIMIT 5;"
```

Or in Python:

```python
import sqlite3, pandas as pd
df = pd.read_sql("SELECT * FROM customers_processed", sqlite3.connect("data/processed/pipeline.db"))
print(df.head())
```
