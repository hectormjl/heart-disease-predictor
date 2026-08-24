"""Reusable inference logic: load the trained pipeline once, and turn one
patient's data into a risk prediction.

Kept separate from api/main.py on purpose — this module has no FastAPI
dependency, so it can be imported and tested (or reused from a CLI script,
a batch job, a notebook, etc.) without spinning up a web server.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_processing import COLUMNS_TO_FLAG_MISSING, PROJECT_ROOT

MODEL_PATH = PROJECT_ROOT / "models" / "model.pkl"
RISK_THRESHOLD = 0.5


def load_model(path: Path = MODEL_PATH):
    """Load the trained pipeline (preprocessing + classifier) from disk.

    This is meant to be called once, when the API process starts (see the
    `lifespan` handler in api/main.py) — not on every request. Unpickling
    the pipeline takes real time; a trained model also doesn't change
    between requests, so there's nothing to gain from reloading it.
    """
    return joblib.load(path)


def patient_to_dataframe(patient: dict) -> pd.DataFrame:
    """Turn one patient's raw input into the single-row DataFrame the
    trained pipeline expects.

    The trained pipeline was fit on data shaped exactly like the output of
    `data_processing.clean_data` — so a live request has to be reshaped the
    same way before it can go through `model.predict_proba`:

    1. `sex`/`fbs`/`exang` get mapped from human-friendly values
       ("male"/"female", True/False) to the 0/1 encoding used at training
       time (mirrors `encode_binary_columns`).
    2. For `ca`/`thal`/`slope` — the fields that need a specialized test the
       patient may not have had — a `<field>_missing` flag is set to 1 when
       the caller left it out, and the value itself becomes NaN so the
       pipeline's imputer fills it in exactly like it did during training
       (mirrors `add_missing_indicators`).
    """
    row = dict(patient)

    row["sex"] = 1 if row["sex"] == "male" else 0
    row["fbs"] = int(row["fbs"])
    row["exang"] = int(row["exang"])

    for col in COLUMNS_TO_FLAG_MISSING:
        is_missing = row.get(col) is None
        row[f"{col}_missing"] = int(is_missing)
        if is_missing:
            row[col] = np.nan

    return pd.DataFrame([row])


def predict_risk(model, patient: dict) -> dict:
    """Run one patient through the pipeline and return probability + label.

    `predict_proba` returns a probability for each class; column 1 is the
    probability of `target == 1` ("disease"), the one we actually care
    about. We threshold it at 0.5 for the label, but the raw probability is
    always returned too — a caller who wants a more conservative screen
    (catch more true positives, accept more false alarms) can apply their
    own, lower threshold to it instead of relying on ours.
    """
    X = patient_to_dataframe(patient)
    probability = float(model.predict_proba(X)[0, 1])
    risk_class = "high_risk" if probability >= RISK_THRESHOLD else "low_risk"
    return {"risk_probability": probability, "risk_class": risk_class}
