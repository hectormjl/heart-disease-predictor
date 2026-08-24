"""Pydantic models defining the API's request and response shapes.

Field constraints double as documentation: FastAPI turns them into an
interactive schema in the /docs Swagger UI, and any value outside these
bounds is rejected with a 422 before it ever reaches the model.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class PatientInput(BaseModel):
    """One patient's clinical data, as collected during a routine exam.

    `slope`, `ca`, and `thal` are optional: they come from specialized
    cardiac tests (exercise ECG, fluoroscopy, a thallium stress test) that
    aren't run at every visit. The EDA behind this project found these
    three fields missing for 30-65% of the training data, largely because
    some clinics in the source dataset simply never ran those tests. Making
    them optional here mirrors that reality instead of forcing the caller
    to fabricate a value. When left out, the trained pipeline imputes them
    the same way it learned to handle missing values during training (see
    `src/predict.py` and `src/data_processing.py`).
    """

    age: int = Field(..., ge=1, le=120, description="Age in years")
    sex: Literal["male", "female"]
    cp: Literal["typical angina", "atypical angina", "non-anginal", "asymptomatic"] = Field(
        ..., description="Chest pain type"
    )
    trestbps: float = Field(..., ge=60, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: bool = Field(..., description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal["normal", "lv hypertrophy", "st-t abnormality"] = Field(
        ..., description="Resting electrocardiogram results"
    )
    thalch: float = Field(..., ge=60, le=250, description="Maximum heart rate achieved")
    exang: bool = Field(..., description="Exercise-induced angina")
    oldpeak: float = Field(
        ..., ge=-5, le=10, description="ST depression induced by exercise, relative to rest"
    )
    slope: Optional[Literal["upsloping", "flat", "downsloping"]] = Field(
        default=None, description="Slope of the peak exercise ST segment (needs a stress test)"
    )
    ca: Optional[int] = Field(
        default=None,
        ge=0,
        le=3,
        description="Number of major vessels colored by fluoroscopy (needs fluoroscopy)",
    )
    thal: Optional[Literal["normal", "fixed defect", "reversable defect"]] = Field(
        default=None, description="Thallium stress test result (needs that test)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63,
                "sex": "male",
                "cp": "typical angina",
                "trestbps": 145,
                "chol": 233,
                "fbs": True,
                "restecg": "lv hypertrophy",
                "thalch": 150,
                "exang": False,
                "oldpeak": 2.3,
                "slope": "downsloping",
                "ca": 0,
                "thal": "fixed defect",
            }
        }
    }


class PredictionResponse(BaseModel):
    risk_probability: float = Field(
        ..., ge=0, le=1, description="Predicted probability of heart disease, 0-1"
    )
    risk_class: Literal["low_risk", "high_risk"] = Field(
        ..., description="risk_probability thresholded at 0.5"
    )
    disclaimer: str = Field(
        default="This is a portfolio project, not a medical device. It does not diagnose "
        "disease and must not be used to make real clinical decisions.",
        description="Always included — this tool is for demonstration purposes only.",
    )


class HealthResponse(BaseModel):
    # `model_config` below disables pydantic's "model_*" protected-namespace
    # check — by default it warns because field names starting with
    # "model_" usually collide with BaseModel's own internal methods, but
    # here they're just plain data fields describing our ML model, so the
    # warning doesn't apply.
    model_config = {"protected_namespaces": ()}

    status: Literal["ok"]
    model_loaded: bool
    model_name: str
