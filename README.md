# Heart Disease Risk Predictor

A small end-to-end ML project: clean a clinical dataset, train and compare a
few classifiers, and serve the best one behind a FastAPI service that
predicts the probability a patient has heart disease from routine exam data.

**This is a portfolio/educational project, not a medical device.** It does
not diagnose disease and must not be used to make real clinical decisions.

## Why this problem

Heart disease is one of the leading causes of death worldwide, and a lot of
the early warning signs live in data that's already collected during a
routine checkup — age, blood pressure, cholesterol, resting ECG, whether
exercise triggers chest pain — without needing an expensive specialized test.
A model that turns those routine numbers into a risk estimate is a plausible
building block for a triage or screening tool: something that helps flag
"this patient probably deserves a closer look" using data a clinic already
has on hand, rather than a tool that replaces a doctor's judgment.

The dataset is the classic combined **UCI Heart Disease dataset**: 920
patients pooled from four sites (Cleveland, Hungary, Switzerland, and VA Long
Beach), each with the same 13 clinical features and a diagnosis label.

## Methodology

**1. EDA** (`notebooks/01_eda.ipynb`)
Explored feature distributions, class balance, and — importantly — missing
data. Three fields (`ca`, `thal`, `slope`) come from specialized cardiac
tests that weren't run at every site, so they're missing for 34-66% of rows,
and that missingness is tied to *which hospital* the patient came from
rather than being random. Also caught a data quality issue: `chol` and
`trestbps` had a bunch of `0` values, which is clinically impossible
(nobody has 0 cholesterol) — those are missing-value placeholders, not real
readings.

**2. Data processing** (`src/data_processing.py`)
- Replaced the placeholder zeros with proper missing values.
- Added an explicit `<field>_missing` flag for `ca`/`thal`/`slope`, so the
  model can use "this wasn't measured" as real information instead of
  silently pretending every imputed value is a genuine reading.
- Collapsed the original 5-level severity target (`num`, 0-4) into a binary
  `target` (0 = no disease, 1 = disease present) — a yes/no risk score is
  the more useful product here, and it's what the API returns.
- Split into train/test (80/20), stratified on the target so both sets keep
  the same ~55/45 class balance.
- All scaling/imputation/encoding is done inside a scikit-learn
  `ColumnTransformer`, fit only on the training data — this avoids leaking
  information from the test set (or, later, from a live API request) into
  how the data gets transformed.

**3. Model training** (`src/train.py`)
Compared three candidates — Logistic Regression, Random Forest, XGBoost —
using 5-fold stratified cross-validation, ranked by ROC-AUC (recall was
tracked too, since in a screening context a missed disease case is worse
than a false alarm). The winner was tuned with a small grid search, then
evaluated once on the held-out test set. The fitted pipeline (preprocessing
+ classifier) is saved to `models/model.pkl`.

**4. API** (`api/`, `src/predict.py`)
A FastAPI service that loads the trained pipeline once at startup and
exposes `POST /predict`. Pydantic validates every request against realistic
field constraints (age ranges, known category values, etc.) before it ever
reaches the model, so bad input gets an automatic 422 with a clear
explanation instead of a confusing error deep in the model code.

**5. Testing** (`tests/`)
Pytest tests for `/predict` covering both valid requests (with and without
the optional specialized-test fields) and invalid ones (missing fields,
out-of-range values, unknown categories, wrong types).

**6. Containerization** (`Dockerfile`)
Packages the API and the trained model into a single Docker image so it can
run anywhere without a manual Python setup.

## How to run

### Locally

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The raw dataset isn't committed to this repo (see Limitations). Place it at
`data/raw/heart_disease_uci.csv` — it needs an `id, age, sex, dataset, cp,
trestbps, chol, fbs, restecg, thalch, exang, oldpeak, slope, ca, thal, num`
column layout (the standard combined UCI Heart Disease CSV). Then:

```bash
python src/data_processing.py   # cleans the data, writes data/processed/{train,test}.csv
python src/train.py             # compares models, tunes the winner, saves models/model.pkl
uvicorn api.main:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive Swagger UI.

### With Docker

`models/model.pkl` has to already exist locally (run `python src/train.py`
first, as above) — `docker build` copies it from your machine, it doesn't
train the model itself.

```bash
docker build -t heart-disease-predictor .
docker run -p 8000:8000 heart-disease-predictor
```

Then hit the same `http://127.0.0.1:8000/docs`.

## Model metrics

Logistic Regression won the model comparison (5-fold CV, ranked by
ROC-AUC) and was the one tuned and saved. Final numbers on the held-out
test set (`models/metrics.json`):

| Metric    | Score |
|-----------|-------|
| Accuracy  | 0.821 |
| Precision | 0.811 |
| Recall    | 0.882 |
| F1        | 0.845 |
| ROC-AUC   | 0.903 |

Recall (0.882) mattered most in picking the final model: in a screening
tool, a false negative — telling a sick patient they're fine — is more
costly than a false positive that just prompts an unnecessary follow-up
test.

## Limitations

- **Small dataset.** ~920 patients across 4 sites is enough to demonstrate
  the pipeline end-to-end, but too small to be confident the model
  generalizes to a broader population.
- **Missingness tied to hospital site.** The `ca`/`thal`/`slope` missing-flags
  carry real signal in this dataset, but because *which fields are missing*
  correlates with which hospital collected the data, the model may be
  partly learning "which site" rather than pure clinical risk — something
  that might not hold for a new hospital's patients.
- **Old data.** The underlying records date back to the late 1980s/early
  1990s; diagnostic practices and patient demographics have moved on since
  then.
- **Binary target loses information.** Collapsing severity grades 1-4 into a
  single "disease present" class is simpler and matches the product need
  here, but throws away how severe the disease is.
- **No calibration check.** The predicted probability is used as-is;
  nothing verifies that "70% risk" predictions are actually right about 70%
  of the time.
- **Not clinically validated.** No external validation, no clinical trial,
  no regulatory review — this is a demonstration of the ML/engineering
  workflow, not a validated screening tool.

## What I'd do differently with more time or data

- Use a larger, more recent, more geographically diverse dataset — ideally
  pooled across more hospitals so the model isn't implicitly learning
  site-specific patterns.
- Calibrate the predicted probabilities (e.g. Platt scaling or isotonic
  regression) so "70% risk" is a number that can actually be trusted at
  face value.
- Add an explainability layer (e.g. SHAP values) so a per-patient
  prediction comes with "here's what drove this score," which matters a lot
  more in a clinical context than in most ML applications.
- Look beyond ROC-AUC/recall at the full precision-recall trade-off, and
  pick an operating threshold deliberately instead of defaulting to 0.5.
- Handle the missing `ca`/`thal`/`slope` data with something more
  principled than single median/mode imputation, e.g. multiple imputation.
- Add a CI pipeline that runs the tests and builds the Docker image on
  every push, plus basic monitoring for input drift if this were ever
  actually deployed.
