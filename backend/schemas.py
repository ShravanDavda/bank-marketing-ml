"""
Pydantic Request and Response Schemas for Bank Marketing Prediction API.
Strictly excludes 'duration' feature per academic problem formulation.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


VALID_JOBS = [
    "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
    "retired", "self-employed", "services", "student", "technician",
    "unemployed", "unknown"
]

VALID_MARITAL = ["divorced", "married", "single"]
VALID_EDUCATION = ["primary", "secondary", "tertiary", "unknown"]
VALID_BINARY = ["no", "yes"]
VALID_CONTACT = ["cellular", "telephone", "unknown"]
VALID_MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
VALID_POUTCOME = ["failure", "other", "success", "unknown"]


class CustomerInputSchema(BaseModel):
    """
    Schema for customer feature input matching the trained model pipeline.
    Duration is intentionally omitted to support pre-contact prediction.
    """
    age: int = Field(..., ge=18, le=100, description="Customer age in years (18 to 100)")
    job: str = Field(..., description="Job occupation category")
    marital: str = Field(..., description="Marital status")
    education: str = Field(..., description="Education level")
    default: str = Field(..., description="Has credit in default ('yes' or 'no')")
    balance: float = Field(..., description="Average yearly balance in Euros (can be negative)")
    housing: str = Field(..., description="Has housing loan ('yes' or 'no')")
    loan: str = Field(..., description="Has personal loan ('yes' or 'no')")
    contact: str = Field(..., description="Contact communication type")
    day: int = Field(..., ge=1, le=31, description="Last contact day of the month")
    month: str = Field(..., description="Last contact month of the year (3-letter abbreviation)")
    campaign: int = Field(..., ge=1, le=65, description="Number of contacts during this campaign")
    pdays: int = Field(..., ge=-1, le=999, description="Days passed since previous campaign contact (-1 = not contacted)")
    previous: int = Field(..., ge=0, le=100, description="Number of contacts performed before this campaign")
    poutcome: str = Field(..., description="Outcome of previous marketing campaign")

    @field_validator("job")
    @classmethod
    def validate_job(cls, v: str) -> str:
        clean_v = v.strip().lower()
        if clean_v not in VALID_JOBS:
            raise ValueError(f"Invalid job '{v}'. Must be one of: {VALID_JOBS}")
        return clean_v

    @field_validator("marital")
    @classmethod
    def validate_marital(cls, v: str) -> str:
        clean_v = v.strip().lower()
        if clean_v not in VALID_MARITAL:
            raise ValueError(f"Invalid marital status '{v}'. Must be one of: {VALID_MARITAL}")
        return clean_v

    @field_validator("education")
    @classmethod
    def validate_education(cls, v: str) -> str:
        clean_v = v.strip().lower()
        if clean_v not in VALID_EDUCATION:
            raise ValueError(f"Invalid education level '{v}'. Must be one of: {VALID_EDUCATION}")
        return clean_v

    @field_validator("default", "housing", "loan")
    @classmethod
    def validate_binary(cls, v: str) -> str:
        clean_v = v.strip().lower()
        if clean_v not in VALID_BINARY:
            raise ValueError(f"Invalid value '{v}'. Must be 'yes' or 'no'")
        return clean_v

    @field_validator("contact")
    @classmethod
    def validate_contact(cls, v: str) -> str:
        clean_v = v.strip().lower()
        if clean_v not in VALID_CONTACT:
            raise ValueError(f"Invalid contact method '{v}'. Must be one of: {VALID_CONTACT}")
        return clean_v

    @field_validator("month")
    @classmethod
    def validate_month(cls, v: str) -> str:
        clean_v = v.strip().lower()
        if clean_v not in VALID_MONTHS:
            raise ValueError(f"Invalid month '{v}'. Must be one of: {VALID_MONTHS}")
        return clean_v

    @field_validator("poutcome")
    @classmethod
    def validate_poutcome(cls, v: str) -> str:
        clean_v = v.strip().lower()
        if clean_v not in VALID_POUTCOME:
            raise ValueError(f"Invalid poutcome '{v}'. Must be one of: {VALID_POUTCOME}")
        return clean_v


class PredictionResponseSchema(BaseModel):
    """
    Schema for model inference response.
    """
    prediction: Literal["yes", "no"]
    prediction_text: str
    subscription_probability: Optional[float]
    probability_available: bool
    model_name: str
    disclaimer: str


class HealthResponseSchema(BaseModel):
    """
    Schema for health check endpoint.
    """
    status: str
    model_loaded: bool
    model_name: str
    timing_constraint: str


class PipelineStageSchema(BaseModel):
    id: str
    number: int
    title: str
    short_desc: str
    status: str
    what_it_means: str
    why_used: str
    what_was_done: str
    inputs: str
    outputs: str
    code_snippet: str
    key_findings: str
    limitations: str


class PipelineResponseSchema(BaseModel):
    title: str
    subtitle: str
    badge: str
    total_stages: int
    stages: List[PipelineStageSchema]
    model_specifications: Dict[str, Any]


class MetricCardSchema(BaseModel):
    name: str
    value: float
    formatted: str
    dataset_split: str
    status: str
    description: str


class ConfusionMatrixQuadrantSchema(BaseModel):
    quadrant: str
    count: int
    percentage_of_test: float
    interpretation: str


class ConfusionMatrixSchema(BaseModel):
    total_test_samples: int
    class_order: Optional[List[str]] = Field(default=["no", "yes"], description="Documented class order")
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    actual_no_total: int
    actual_yes_total: int
    predicted_no_total: int
    predicted_yes_total: int
    breakdown: List[ConfusionMatrixQuadrantSchema]


class BaselineVsTunedSchema(BaseModel):
    model: str
    hyperparameters: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    notes: str


class BaselineModelSchema(BaseModel):
    model: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: float
    rank_f1: int
    selection_status: str


class FeatureImportanceItemSchema(BaseModel):
    feature: str
    importance: float
    importance_pct: float
    rank: int
    description: str


class ProjectLimitationSchema(BaseModel):
    number: int
    title: str
    detail: str


class PerformanceResponseSchema(BaseModel):
    title: str
    subtitle: str
    test_set_size: int
    test_split_ratio: str
    summary_metrics: List[MetricCardSchema]
    confusion_matrix: ConfusionMatrixSchema
    baseline_vs_tuned: List[BaselineVsTunedSchema]
    all_baseline_models: List[BaselineModelSchema]
    feature_importances: List[FeatureImportanceItemSchema]
    limitations: List[ProjectLimitationSchema]


class DatasetSummaryResponseSchema(BaseModel):
    filename: str
    total_records: int
    total_columns: int
    delimiter: str
    missing_values: int
    duplicate_rows: int
    target_column: str
    target_classes: List[str]
    target_distribution: Dict[str, Any]
    numerical_features_count: int
    categorical_features_count: int
    numerical_features: List[str]
    categorical_features: List[str]
    model_features_count: int
    engineered_features: List[str]
    excluded_features: List[str]
    pdays_negative_one_count: int
    pdays_negative_one_pct: float


class FeatureCatalogueItemSchema(BaseModel):
    name: str
    data_type: str
    role: str
    used_in_model: bool
    description: str
    examples: str
    notes: str


class DatasetFeaturesResponseSchema(BaseModel):
    total_features: int
    features: List[FeatureCatalogueItemSchema]


class DatasetPreviewResponseSchema(BaseModel):
    total_records: int
    total_pages: int
    current_page: int
    page_size: int
    columns: List[str]
    rows: List[Dict[str, Any]]


class EDAResponseSchema(BaseModel):
    target_distribution: List[Dict[str, Any]]
    subscription_rate_by_poutcome: List[Dict[str, Any]]
    subscription_rate_by_contact: List[Dict[str, Any]]
    subscription_rate_by_housing: List[Dict[str, Any]]
    subscription_rate_by_job: List[Dict[str, Any]]
    numerical_comparisons: Dict[str, Any]

