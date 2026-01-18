"""
FastAPI application for Insurance Fraud Detection
Uses CatBoost model with native categorical handling

TO RUN:
  uvicorn api.main:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
import pandas as pd
import joblib
from catboost import CatBoostClassifier
from pathlib import Path
import sys

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent))
from api.schemas import ClaimInput, PredictionResponse, HealthResponse

# Initialize FastAPI app
app = FastAPI(
    title="Insurance Fraud Detection API",
    description="Predict insurance claim fraud using CatBoost ML model",
    version="1.0.0"
)

# Global variables
model = None
feature_names = None
cat_feature_indices = None

# Model path
MODEL_DIR = Path(__file__).parent.parent / "model"


def extract_from_range(value: str) -> float:
    """Extract numerical value from string ranges"""
    if pd.isna(value) or value == 'none':
        return 0.0

    value = str(value).strip()

    # Handle 'more than X' or 'X years'
    if 'more than' in value.lower():
        num = ''.join(filter(str.isdigit, value))
        return float(num) + 15 if num else 0.0

    if 'years' in value.lower() or 'year' in value.lower():
        num = ''.join(filter(str.isdigit, value))
        return float(num) if num else 0.0

    # Handle range like '15-30' or '26 to 30'
    if '-' in value or 'to' in value.lower():
        parts = value.replace('to', '-').split('-')
        try:
            lower = float(''.join(filter(lambda x: x.isdigit() or x == '.', parts[0])))
            upper = float(''.join(filter(lambda x: x.isdigit() or x == '.', parts[1])))
            return (lower + upper) / 2
        except:
            return 0.0

    # Try to extract any number
    try:
        num = ''.join(filter(lambda x: x.isdigit() or x == '.', value))
        return float(num) if num else 0.0
    except:
        return 0.0


def preprocess_claim(claim_data: dict) -> pd.DataFrame:
    """Preprocess claim data for CatBoost (raw categoricals, no scaling)"""

    # Extract numerical values from string ranges
    processed = claim_data.copy()

    processed['PolicyAccidentDays'] = extract_from_range(processed['Days:Policy-Accident'])
    processed['PolicyClaimDays'] = extract_from_range(processed['Days:Policy-Claim'])
    processed['PastClaimsNum'] = extract_from_range(processed['PastNumberOfClaims'])
    processed['VehicleAge'] = extract_from_range(processed['AgeOfVehicle'])
    processed['PolicyHolderAge'] = extract_from_range(processed['AgeOfPolicyHolder'])
    processed['NumSuppliments'] = extract_from_range(processed['NumberOfSuppliments'])
    processed['NumCars'] = extract_from_range(processed['NumberOfCars'])

    # Map VehiclePrice to ordinal
    price_mapping = {
        'less than 20000': 1,
        '20000 to 29000': 2,
        '30000 to 39000': 3,
        '40000 to 59000': 4,
        '60000 to 69000': 5,
        'more than 69000': 6
    }
    processed['VehiclePriceOrdinal'] = price_mapping.get(processed['VehiclePrice'], 3)

    # Create DataFrame
    df = pd.DataFrame([processed])

    # Select features in the EXACT order from training
    # This matches the order from notebook 02
    selected_features = []
    for feat in feature_names:
        if feat in df.columns:
            selected_features.append(feat)
        else:
            # If feature doesn't exist, add with None
            df[feat] = None
            selected_features.append(feat)

    result_df = df[feature_names]

    # Debug: print what we're sending
    print(f"Feature count: {len(feature_names)}")
    print(f"DataFrame shape: {result_df.shape}")

    return result_df


@app.on_event("startup")
async def load_model():
    """Load CatBoost model and preprocessing artifacts on startup"""
    global model, feature_names, cat_feature_indices

    try:
        # Load CatBoost model
        model = CatBoostClassifier()
        model.load_model(str(MODEL_DIR / "catboost_model.cbm"))

        # Load feature names and categorical indices
        feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")
        cat_feature_indices = joblib.load(MODEL_DIR / "cat_feature_indices.pkl")

        print("✓ CatBoost model loaded successfully!")
        print(f"  Features: {len(feature_names)}")
        print(f"  Categorical features: {len(cat_feature_indices)}")

    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        raise


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Insurance Fraud Detection API",
        "model": "CatBoost",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    model_loaded = model is not None

    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_fraud(claim: ClaimInput):
    """
    Predict fraud probability for an insurance claim

    Uses CatBoost model with native categorical feature handling.
    Returns fraud probability, risk score, and confidence level.
    """

    try:
        # Convert to dict
        claim_dict = claim.model_dump(by_alias=True)

        # Preprocess for CatBoost
        X = preprocess_claim(claim_dict)

        # Make prediction (model already knows which features are categorical)
        probability = model.predict_proba(X)[0][1]

        # Calculate outputs
        fraud_prob = float(probability)
        risk_score = int(fraud_prob * 100)
        prediction = "Fraud" if fraud_prob >= 0.5 else "No Fraud"

        # Determine confidence level
        if fraud_prob >= 0.8 or fraud_prob <= 0.2:
            confidence = "High"
        elif fraud_prob >= 0.6 or fraud_prob <= 0.4:
            confidence = "Medium"
        else:
            confidence = "Low"

        return PredictionResponse(
            fraud_probability=round(fraud_prob, 4),
            fraud_risk_score=risk_score,
            prediction=prediction,
            confidence=confidence
        )

    except Exception as e:
        # Print detailed error for debugging
        print(f"Prediction error details: {str(e)}")
        print(f"Input features: {X.columns.tolist()}")
        print(f"Expected features: {feature_names}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


# DON'T run this block if importing in Jupyter
# Only runs when executed directly from command line
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)