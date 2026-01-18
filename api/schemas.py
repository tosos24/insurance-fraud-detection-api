"""
Pydantic schemas for request/response validation
"""
from pydantic import BaseModel, Field, ConfigDict
from typing import Literal


class ClaimInput(BaseModel):
    """Input schema for insurance claim prediction"""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
                "Month": "Dec",
                "WeekOfMonth": 5,
                "DayOfWeek": "Wednesday",
                "Make": "Honda",
                "AccidentArea": "Urban",
                "DayOfWeekClaimed": "Tuesday",
                "MonthClaimed": "Jan",
                "WeekOfMonthClaimed": 1,
                "Sex": "Male",
                "MaritalStatus": "Single",
                "Age": 21,
                "Fault": "Policy Holder",
                "PolicyType": "Sport - Liability",
                "VehicleCategory": "Sport",
                "VehiclePrice": "more than 69000",
                "Days:Policy-Accident": "more than 30",
                "Days:Policy-Claim": "more than 30",
                "PastNumberOfClaims": "none",
                "AgeOfVehicle": "3 years",
                "AgeOfPolicyHolder": "26 to 30",
                "PoliceReportFiled": "Yes",
                "WitnessPresent": "No",
                "AgentType": "External",
                "NumberOfSuppliments": "none",
                "NumberOfCars": "3 to 4",
                "Year": 1994,
                "BasePolicy": "Liability",
                "Deductible": 400,
                "DriverRating": 1
            }
        }
    )

    # Numerical features
    Age: int = Field(..., ge=16, le=100, description="Age of policy holder")
    Deductible: int = Field(..., ge=0, description="Deductible amount")
    DriverRating: int = Field(..., ge=1, le=4, description="Driver rating (1-4)")
    WeekOfMonth: int = Field(..., ge=1, le=5, description="Week of month")
    WeekOfMonthClaimed: int = Field(..., ge=1, le=5, description="Week of month claimed")
    Year: int = Field(..., ge=1990, le=2030, description="Year")

    # Categorical features
    Month: str = Field(..., description="Month of accident")
    DayOfWeek: str = Field(..., description="Day of week")
    Make: str = Field(..., description="Vehicle make")
    AccidentArea: str = Field(..., description="Accident area (Urban/Rural)")
    DayOfWeekClaimed: str = Field(..., description="Day claim was filed")
    MonthClaimed: str = Field(..., description="Month claim was filed")
    Sex: str = Field(..., description="Gender (Male/Female)")
    MaritalStatus: str = Field(..., description="Marital status")
    Fault: str = Field(..., description="Fault (Policy Holder/Third Party)")
    PolicyType: str = Field(..., description="Policy type")
    VehicleCategory: str = Field(..., description="Vehicle category")
    PoliceReportFiled: str = Field(..., description="Police report filed (Yes/No)")
    WitnessPresent: str = Field(..., description="Witness present (Yes/No)")
    AgentType: str = Field(..., description="Agent type")
    BasePolicy: str = Field(..., description="Base policy type")

    # String range features
    Days_Policy_Accident: str = Field(..., alias="Days:Policy-Accident", description="Days between policy and accident")
    Days_Policy_Claim: str = Field(..., alias="Days:Policy-Claim", description="Days between policy and claim")
    PastNumberOfClaims: str = Field(..., description="Past number of claims")
    AgeOfVehicle: str = Field(..., description="Age of vehicle")
    AgeOfPolicyHolder: str = Field(..., description="Age category of policy holder")
    NumberOfSuppliments: str = Field(..., description="Number of supplements")
    NumberOfCars: str = Field(..., description="Number of cars")
    VehiclePrice: str = Field(..., description="Vehicle price range")


class PredictionResponse(BaseModel):
    """Response schema for fraud prediction"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "fraud_probability": 0.78,
                "fraud_risk_score": 78,
                "prediction": "Fraud",
                "confidence": "High"
            }
        }
    )

    fraud_probability: float = Field(..., ge=0, le=1, description="Probability of fraud (0-1)")
    fraud_risk_score: int = Field(..., ge=0, le=100, description="Fraud risk score (0-100)")
    prediction: Literal["Fraud", "No Fraud"] = Field(..., description="Fraud prediction")
    confidence: str = Field(..., description="Confidence level (Low/Medium/High)")


class HealthResponse(BaseModel):
    """Health check response"""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_name": "CatBoost",
                "version": "1.0.0"
            }
        }
    )

    status: str
    model_loaded: bool
    model_name: str = "CatBoost"
    version: str = "1.0.0"