"""
Pydantic schemas for input validation and output formatting.
"""

from pydantic import BaseModel, Field


# ─── Regression ───────────────────────────────────────────────

class RegressionInput(BaseModel):
    """Input features for house price prediction (regression)."""

    GrLivArea: float = Field(..., gt=0, description="Above grade living area (sq ft)")
    TotalBsmtSF: float = Field(..., ge=0, description="Total basement area (sq ft)")
    LotArea: float = Field(..., gt=0, description="Lot size (sq ft)")
    BedroomAbvGr: int = Field(..., ge=0, description="Number of bedrooms above grade")
    FullBath: int = Field(..., ge=0, description="Number of full bathrooms")
    TotRmsAbvGrd: int = Field(..., ge=0, description="Total rooms above grade")
    OverallQual: int = Field(..., ge=1, le=10, description="Overall material and finish quality (1-10)")
    OverallCond: int = Field(..., ge=1, le=10, description="Overall condition rating (1-10)")
    YearBuilt: int = Field(..., ge=1800, le=2030, description="Year built")
    YearRemodAdd: int = Field(..., ge=1800, le=2030, description="Year of remodel")
    Neighborhood: str = Field(..., description="Physical location within Ames city limits")
    GarageCars: int = Field(..., ge=0, description="Garage capacity in cars")
    GarageArea: float = Field(..., ge=0, description="Garage area (sq ft)")
    PoolArea: float = Field(..., ge=0, description="Pool area (sq ft)")
    Fireplaces: int = Field(..., ge=0, description="Number of fireplaces")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "GrLivArea": 1500,
                    "TotalBsmtSF": 800,
                    "LotArea": 9000,
                    "BedroomAbvGr": 3,
                    "FullBath": 2,
                    "TotRmsAbvGrd": 7,
                    "OverallQual": 7,
                    "OverallCond": 5,
                    "YearBuilt": 2000,
                    "YearRemodAdd": 2005,
                    "Neighborhood": "NAmes",
                    "GarageCars": 2,
                    "GarageArea": 500,
                    "PoolArea": 0,
                    "Fireplaces": 1,
                }
            ]
        }
    }


class RegressionOutput(BaseModel):
    """Output for house price prediction."""

    model_used: str = Field(..., description="Name of the model used for prediction")
    predicted_price: float = Field(..., description="Predicted sale price")
    currency: str = Field(default="USD", description="Currency of the predicted price")


# ─── Classification ───────────────────────────────────────────

class ClassificationInput(BaseModel):
    """Input features for building type classification."""

    GrLivArea: float = Field(..., gt=0, description="Above grade living area (sq ft)")
    TotRmsAbvGrd: int = Field(..., ge=0, description="Total rooms above grade")
    OverallQual: int = Field(..., ge=1, le=10, description="Overall material and finish quality (1-10)")
    YearBuilt: int = Field(..., ge=1800, le=2030, description="Year built")
    GarageCars: int = Field(..., ge=0, description="Garage capacity in cars")
    Neighborhood: str = Field(..., description="Physical location within Ames city limits")
    HouseStyle: str = Field(..., description="Style of dwelling (e.g. 1Story, 2Story)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "GrLivArea": 1500,
                    "TotRmsAbvGrd": 7,
                    "OverallQual": 7,
                    "YearBuilt": 2000,
                    "GarageCars": 2,
                    "Neighborhood": "NAmes",
                    "HouseStyle": "1Story",
                }
            ]
        }
    }


class ClassificationOutput(BaseModel):
    """Output for building type classification."""

    model_used: str = Field(..., description="Name of the model used for prediction")
    predicted_type: str = Field(..., description="Predicted building type label")
    predicted_type_encoded: int = Field(..., description="Encoded value of the predicted type")
