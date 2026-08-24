"""Data cleaning, feature engineering, and train/test splitting for the
Heart Disease UCI dataset.

This module turns the raw CSV into model-ready data. It does NOT scale or
one-hot encode anything itself - that logic lives in a scikit-learn
ColumnTransformer (see `build_preprocessing_pipeline`) so it can be fit only
on the training set and reused, unchanged, at inference time in the API.
Baking scaled/encoded values into a static CSV would risk train/serve skew:
the API would need to reimplement the exact same transform by hand.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "heart_disease_uci.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# --- Feature groups (columns as they exist AFTER clean_data) ---
# Kept as named constants instead of scattering string literals through the
# code, so train.py can import the exact same groups the pipeline expects.
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak", "ca"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]
# These three (not chol, despite it also having a lot of NaNs after the
# zero-placeholder fix below) get an explicit "was this value missing" flag.
# Reason: the EDA showed ca/thal/slope missingness is driven by *which
# hospital* ran which tests, a real, structural pattern worth flagging.
# chol's missingness is a data-entry artifact (0-as-placeholder), a
# different kind of problem that a "missing" flag wouldn't meaningfully
# explain - so it's just imputed like any other numeric column.
COLUMNS_TO_FLAG_MISSING = ["ca", "thal", "slope"]
INDICATOR_FEATURES = [f"{col}_missing" for col in COLUMNS_TO_FLAG_MISSING]
TARGET_COLUMN = "target"


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as downloaded, no modifications."""
    return pd.read_csv(path)


def fix_placeholder_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """Replace clinically impossible 0 values with NaN.

    The EDA found ~170 rows where `chol` (cholesterol) is exactly 0, and one
    row where `trestbps` (resting blood pressure) is 0. Nobody has 0
    cholesterol or 0 blood pressure while alive - these are missing-value
    placeholders, not real measurements. Left as-is, the model would learn
    a nonsensical "0 is a valid cholesterol reading" pattern.
    """
    df = df.copy()
    df.loc[df["chol"] == 0, "chol"] = pd.NA
    df.loc[df["trestbps"] == 0, "trestbps"] = pd.NA
    return df


def encode_binary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map human-readable binary columns to 0/1, keeping NaN as NaN.

    `sex` arrives as "Male"/"Female" and `fbs`/`exang` arrive as Python
    True/False (parsed that way by pandas from the CSV's TRUE/FALSE text).
    Mapping them to 0/1 here - rather than inside the ColumnTransformer -
    keeps the pipeline's job focused on imputation/scaling/encoding, not
    bookkeeping conversions.
    """
    df = df.copy()
    df["sex"] = df["sex"].map({"Male": 1, "Female": 0})
    df["fbs"] = df["fbs"].map({True: 1, False: 0})
    df["exang"] = df["exang"].map({True: 1, False: 0})
    return df


def add_missing_indicators(
    df: pd.DataFrame, columns: list[str] = COLUMNS_TO_FLAG_MISSING
) -> pd.DataFrame:
    """Add a `<col>_missing` binary flag for `columns`, before they get
    imputed elsewhere in the pipeline.

    Why this matters here specifically: `ca` (66% missing), `thal` (53%),
    and `slope` (34%) aren't missing at random - the EDA showed missingness
    is tied to *which hospital* collected the data. Imputing that much of a
    column (median/mode) silently invents a large share of its values.
    Adding an explicit "was this originally missing" flag at least lets the
    model use that information honestly instead of pretending every
    imputed value is a real reading.

    Caveat worth remembering (documented in the README's limitations
    section): because missingness correlates with hospital site, these
    flags can also act as a proxy for "which site this patient came from" -
    a real relationship in this dataset, but not something that will
    necessarily generalize to a new hospital's patients.
    """
    df = df.copy()
    for col in columns:
        df[f"{col}_missing"] = df[col].isnull().astype(int)
    return df


def binarize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the 5-level severity target `num` (0-4) into binary `target`.

    0 stays "no disease" (0); grades 1-4 (increasing severity) all become
    "disease present" (1). A risk predictor answering "yes/no, and how
    confident" is the more useful product than a 5-way severity classifier,
    and it matches how the API will present results (see Phase 4).
    """
    df = df.copy()
    df[TARGET_COLUMN] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline and return model-ready (but not yet
    scaled/encoded) data.

    Two columns are dropped entirely rather than cleaned:
    - `id`: a row identifier, carries no clinical information.
    - `dataset`: which hospital/site the record came from. It was useful
      during EDA (it explained the missingness patterns above), but a real
      patient using the API has no "UCI collection site" to report. Keeping
      it as a feature would let the model quietly learn site-specific base
      rates instead of genuine clinical relationships - a form of leakage
      that would make the model impossible to use correctly in production.
    """
    df = fix_placeholder_zeros(df)
    df = encode_binary_columns(df)
    df = add_missing_indicators(df)
    df = binarize_target(df)
    df = df.drop(columns=["id", "dataset"])
    return df


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Build the (unfit) preprocessing step shared by every model in train.py.

    This does NOT touch the data - it only defines *how* each feature group
    should be transformed once it's fit on the training set:

    - Numeric features: impute missing values with the column median (robust
      to the outliers we saw in the EDA), then scale to mean 0 / std 1.
      Scaling matters for Logistic Regression (its regularization penalizes
      large coefficients, so features need comparable ranges) and is
      harmless for tree-based models like Random Forest/XGBoost.
    - Binary features (already 0/1): impute missing values with the most
      frequent value (mode). No scaling needed - they're already on a 0-1
      range.
    - Categorical features: impute missing values with the mode, then
      one-hot encode. `handle_unknown="ignore"` means a category never seen
      during training (e.g. a typo, or a new hospital's naming convention)
      produces all-zero dummy columns at inference instead of crashing.
    - Missing-indicator flags: already clean 0/1 columns, passed through
      unchanged.

    Returns an unfit ColumnTransformer - train.py fits it (as part of a
    full Pipeline, together with a classifier) on the training data only,
    then reuses that exact fitted transform for the test set and, later,
    for live API requests.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    binary_transformer = SimpleImputer(strategy="most_frequent")

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ("binary", binary_transformer, BINARY_FEATURES),
        ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
        ("indicator", "passthrough", INDICATOR_FEATURES),
    ])


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split, keeping the target column in both halves.

    Stratifying on `target` preserves the ~55/45 class balance found in the
    EDA in both splits - without it, a small/unlucky split could end up
    more imbalanced than the full dataset by chance. `random_state` is
    fixed so the split is reproducible: anyone re-running this script gets
    the exact same train/test rows.
    """
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COLUMN],
    )  # sklearn's stubs type this generically; both are DataFrames at runtime
    return train_df, test_df  # type: ignore[return-value]


if __name__ == "__main__":
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    train_df, test_df = split_data(clean_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print(f"Train shape: {train_df.shape}, test shape: {test_df.shape}")
    print("Train target balance:\n", train_df[TARGET_COLUMN].value_counts(normalize=True).round(3))
    print("Test target balance:\n", test_df[TARGET_COLUMN].value_counts(normalize=True).round(3))
