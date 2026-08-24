"""Train, compare, tune, and persist a heart disease risk classifier.

Pipeline: load raw data -> clean it (data_processing.py) -> compare a few
candidate models with cross-validation -> tune the winner's hyperparameters
-> do a single, final evaluation on the untouched test set -> save the
fitted model + its metrics.
"""

import json
from datetime import datetime, timezone

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data_processing import (
    PROJECT_ROOT,
    TARGET_COLUMN,
    build_preprocessing_pipeline,
    clean_data,
    load_raw_data,
    split_data,
)

MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42

# Metrics tracked for every model. `roc_auc` (area under the ROC curve) is
# what we use to RANK candidates: it measures how well a model separates
# the two classes across every possible decision threshold, so it doesn't
# depend on the default 0.5 cutoff. We still track `recall` for every
# model because it's what actually matters for this use case: in a health
# screening context, a false negative (telling a sick patient they're fine)
# is far more costly than a false positive (flagging a healthy patient for
# a follow-up test that then comes back clear). Accuracy alone can hide a
# model that's quietly bad at catching the positive class.
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]

CANDIDATE_MODELS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "xgboost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
}

# Small hyperparameter grids, applied only to whichever model wins the
# comparison below. Kept deliberately modest: with ~700 training rows, an
# exhaustive search would just start tuning to noise instead of signal.
#
# - LogisticRegression's `C` controls regularization strength (inverse of
#   it, technically): smaller C = stronger penalty on large coefficients =
#   simpler, more conservative model.
# - RandomForest's `max_depth`/`min_samples_leaf` control how complex each
#   tree is allowed to get; shallower/more-restricted trees resist
#   overfitting on a small dataset.
# - XGBoost's `learning_rate` scales how much each boosting round corrects
#   the previous one; lower rates need more `n_estimators` (rounds) to
#   reach the same fit, but tend to generalize better.
PARAM_GRIDS = {
    "logistic_regression": {
        "classifier__C": [0.01, 0.1, 1, 10],
    },
    "random_forest": {
        "classifier__n_estimators": [100, 300],
        "classifier__max_depth": [3, 5, None],
        "classifier__min_samples_leaf": [1, 3],
    },
    "xgboost": {
        "classifier__n_estimators": [100, 300],
        "classifier__max_depth": [2, 3, 4],
        "classifier__learning_rate": [0.05, 0.1],
    },
}


def build_model_pipeline(classifier) -> Pipeline:
    """Wrap the shared preprocessing step around one classifier.

    Bundling preprocessing + model into a single Pipeline means every fold
    of cross-validation refits imputation/scaling/encoding using only that
    fold's training rows — the same leakage risk we avoided earlier between
    train and test, just one level deeper (between CV folds).
    """
    return Pipeline(steps=[
        ("preprocessing", build_preprocessing_pipeline()),
        ("classifier", classifier),
    ])


def compare_models(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Cross-validate every candidate model and return a comparison table.

    5-fold stratified cross-validation: the training data is split into 5
    folds; each model trains on 4 of them and is scored on the 1 held out,
    rotating which fold is held out each time. "Stratified" keeps the
    ~55/45 class balance consistent in every fold. Averaging over 5 fits
    gives a far more reliable performance estimate than a single
    train/validation split — and its standard deviation across folds shows
    how *stable* that estimate is.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []

    for name, classifier in CANDIDATE_MODELS.items():
        pipeline = build_model_pipeline(classifier)
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=SCORING)
        row = {"model": name}
        for metric in SCORING:
            row[metric] = scores[f"test_{metric}"].mean()
            row[f"{metric}_std"] = scores[f"test_{metric}"].std()
        rows.append(row)

    results = pd.DataFrame(rows).set_index("model")
    return results.sort_values("roc_auc", ascending=False)


def tune_best_model(best_name: str, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Grid-search the winning model's hyperparameters.

    GridSearchCV trains and scores every combination in
    `PARAM_GRIDS[best_name]` using the same 5-fold cross-validation as
    `compare_models`, then automatically refits the best-scoring
    combination on the *entire* training set — that refit is what
    `.best_estimator_` gives back, no separate manual refit step needed.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipeline = build_model_pipeline(CANDIDATE_MODELS[best_name])
    grid_search = GridSearchCV(
        pipeline,
        param_grid=PARAM_GRIDS[best_name],
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)
    print(f"Best params for {best_name}: {grid_search.best_params_}")
    print(f"Best CV ROC-AUC after tuning: {grid_search.best_score_:.3f}")
    return grid_search.best_estimator_


def evaluate_on_test(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """One final, one-time check on data the model has never touched in any
    form — not during training, not during cross-validation. This is the
    number that best represents how the model would perform on a new
    patient.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    print("\nTest set classification report:")
    print(classification_report(y_test, y_pred, target_names=["no disease", "disease"]))
    print("Confusion matrix (rows = actual, columns = predicted):")
    print(confusion_matrix(y_test, y_pred))

    return metrics


def main() -> None:
    raw_df = load_raw_data()
    clean_df = clean_data(raw_df)
    train_df, test_df = split_data(clean_df)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    print("Comparing candidate models with 5-fold cross-validation...\n")
    comparison = compare_models(X_train, y_train)
    print(comparison.round(3)[SCORING])

    best_name = comparison.index[0]
    print(f"\nBest model by cross-validated ROC-AUC: {best_name}")

    print(f"\nTuning {best_name} with GridSearchCV...")
    best_model = tune_best_model(best_name, X_train, y_train)

    test_metrics = evaluate_on_test(best_model, X_test, y_test)
    print("\nFinal test set metrics:", {k: round(v, 3) for k, v in test_metrics.items()})

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "model.pkl")

    metadata = {
        "model_name": best_name,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "cv_comparison": comparison.round(4).reset_index().to_dict(orient="records"),
        "test_metrics": test_metrics,
    }
    with open(MODELS_DIR / "metrics.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model to {MODELS_DIR / 'model.pkl'}")
    print(f"Saved metrics to {MODELS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
