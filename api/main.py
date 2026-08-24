"""FastAPI application serving the heart disease risk model.

Run locally from the project root with:
    uvicorn api.main:app --reload

Then open http://127.0.0.1:8000/docs for the interactive Swagger UI.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from api.schemas import HealthResponse, PatientInput, PredictionResponse
from src.predict import load_model, predict_risk

# Plain dict holding process-wide state (the loaded model). This is enough
# for a single-model, single-process portfolio API — no need to reach for
# a database or cache for something this small.
MODEL_STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model exactly once, when the server process starts.

    FastAPI runs the code before `yield` on startup and the code after it
    on shutdown. Loading here — instead of inside the /predict endpoint —
    means the (relatively slow) joblib.load only happens once, not on
    every request.
    """
    MODEL_STATE["model"] = load_model()
    MODEL_STATE["model_name"] = type(MODEL_STATE["model"].named_steps["classifier"]).__name__
    yield
    MODEL_STATE.clear()


app = FastAPI(
    title="Heart Disease Risk Predictor",
    description=(
        "Predicts the probability that a patient has heart disease from "
        "routine clinical data. Portfolio/educational project — NOT a "
        "medical diagnostic tool."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {"message": "Heart Disease Risk Predictor API. See /docs for usage."}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness/readiness check: is the process up, and is the model
    actually loaded and ready to serve predictions?
    """
    return HealthResponse(
        status="ok",
        model_loaded="model" in MODEL_STATE,
        model_name=MODEL_STATE.get("model_name", "not_loaded"),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput) -> PredictionResponse:
    """Predict heart disease risk for one patient.

    FastAPI validates the request body against `PatientInput` before this
    function runs at all — an out-of-range age, an unrecognized chest pain
    type, a missing required field, etc. never reaches this code. The
    caller gets an automatic 422 response with a field-by-field explanation
    of what was wrong.
    """
    model = MODEL_STATE.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    result = predict_risk(model, patient.model_dump())
    return PredictionResponse(**result)
