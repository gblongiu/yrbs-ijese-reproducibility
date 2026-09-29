"""Configuration used by the article reproduction workflow."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUTS_DIR / "tables"
LOGS_DIR = OUTPUTS_DIR / "logs"

RAW_FILE_2023 = RAW_DIR / "YRBS_2023_MH_subset.xlsx"

TARGET_PRIMARY = "QN26"
BULLYING_EXPOSURES = ["QN24", "QN25"]
SECONDARY_TARGETS = ["QN27", "QN28", "QN29", "QN30"]

# Dataset-build configuration (source-column names).
# Baseline model = demographic covariates only.
# Bullying-augmented model = baseline covariates plus QN24 and QN25.
BASELINE_COVARIATES = ["q1", "q2", "q3", "raceeth"]

# Survey design fields are used for weighted descriptive summaries only (not as model features).
SURVEY_DESIGN_COLS = ["weight", "stratum", "psu"]

TARGET_COL = "y_qn26"
TEST_SIZE = 0.2
CV_FOLDS = 5
