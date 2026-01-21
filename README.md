 Insurance Claim Fraud Detection API



A machine learning system for detecting fraudulent insurance claims using CatBoost with native categorical feature handling, deployed as a REST API.



---



 Project Overview



This project implements an end-to-end machine learning pipeline for insurance fraud detection, from exploratory data analysis to a production-ready API. The system analyzes 33 features from insurance claims to predict fraud probability with 85.4% accuracy (ROC-AUC).



Key Highlights:

\- 85.4% ROC-AUC on test set using CatBoost

\- Dual preprocessing pipelines optimized for different model types

\- Native categorical handling - no one-hot encoding needed for CatBoost

\- FastAPI deployment with automatic documentation

\- Comprehensive model comparison (3 algorithms tested)



---



 Model Performance



 Model Comparison:



| Model | Validation AUC | Test AUC | Training Time |

|-------|---------------|----------|---------------|

| CatBoost (Winner) | 0.8693 | 0.8538 | ~45s |

| Logistic Regression | 0.8083 | 0.7878 | ~3s |

| PyTorch MLP | 0.7805 | 0.7555 | ~120s |



Winner: CatBoost with 85.38% ROC-AUC on test set



 Why CatBoost Won:



1\. Native categorical handling - Preserves information from high-cardinality features like vehicle make

2\. No feature scaling needed - Tree-based algorithm handles raw numerical data

3\. Better generalization - 1.6% higher AUC than Logistic Regression



---



 Key Insights



 Most Important Features for Fraud Detection:



1\. Fault (by far the strongest predictor)

&nbsp;  - Policy holder at fault vs. third party

&nbsp;  - Dominant feature with significant gap to others



2\. Base Policy

&nbsp;  - Type of insurance policy held
  

3\. Year

&nbsp;  - Claim year



4\. Other notable features:

&nbsp;  - Policy Type

&nbsp;  - Deductible

&nbsp;  - Past Number of Claims

&nbsp;  - Vehicle Category



 Data Insights:



\- Dataset: 15,420 insurance claims with 33 features

\- Class Distribution: Imbalanced (handled with SMOTE)

\- Feature Types: Mixed categorical and numerical

\- Preprocessing Strategy: Model-specific pipelines for optimal performance



---



 Project Structure



```

insurance-fraud-detection/

│

├── data/

│   └── carclaims.csv                  # Original dataset (15,420 records)

│

├── notebooks/

│   ├── 01\_eda.ipynb                   # Exploratory Data Analysis

│   ├── 02\_feature\_engineering.ipynb   # Dual preprocessing pipelines

│   └── 03\_model\_training.ipynb        # Model training \& comparison

│

├── api/

│   ├── main.py                        # FastAPI application

│   └── schemas.py                     # Request/response validation

│

├── saved\_models/

│   ├── catboost\_model.cbm             # Trained CatBoost model

│   ├── logistic\_regression.pkl        # Trained LR model

│   ├── mlp\_model.pth                  # Trained PyTorch MLP

│   ├── scaler.pkl                     # Feature scaler (LR/MLP only)

│   ├── feature\_names\_catboost.pkl     # Feature names for CatBoost

│   ├── catboost\_cat\_indices.pkl       # Categorical feature indices

│   └── model\_comparison.csv           # Performance comparison

│

├── requirements.txt                    # Python dependencies

└── README.md                           # This file

```



---



 Quick Start



 Prerequisites:



\- Python 3.10+

\- pip



 Installation:



1\. Clone the repository

```bash

git clone https://github.com/yourusername/insurance-fraud-detection.git

cd insurance-fraud-detection

```



2\. Install dependencies

```bash

pip install -r requirements.txt

```



3\. Run the API

```bash

uvicorn api.main:app --reload

```



4\. Access the API

\- Interactive docs: http://localhost:8000/docs

\- Health check: http://localhost:8000/health



---



 API Usage



 Make a Prediction:



Endpoint: POST /predict



Example Request:

```bash

curl -X POST "http://localhost:8000/predict" \\

 -H "Content-Type: application/json" \\

 -d '{

   "Month": "Dec",

   "WeekOfMonth": 5,

   "DayOfWeek": "Wednesday",

   "Make": "Honda",

   "AccidentArea": "Urban",

   "Sex": "Male",

   "Age": 21,

   "Fault": "Policy Holder",

   "PolicyType": "Sport - Liability",

   "VehicleCategory": "Sport",

   "VehiclePrice": "more than 69000",

   "PoliceReportFiled": "Yes",

   "WitnessPresent": "No",

   "PastNumberOfClaims": "none",

   "AgeOfVehicle": "3 years",

   ... (33 fields total)

 }'

```



Example Response:

```json

{

 "fraud_probability": 0.78,

 "fraud_risk_score": 78,

 "prediction": "Fraud",

 "confidence": "High"

}

```



 Risk Score Interpretation:



\- 0-30: Low risk - Claim appears legitimate

\- 31-60: Medium risk - Further review recommended

\- 61-100: High risk - Likely fraudulent claim



---



 Technical Approach



 Dual Preprocessing Pipelines:



This project implements model-specific preprocessing for optimal performance:



Pipeline A: CatBoost (Optimized)

\- Keeps categorical features as raw strings

\- No one-hot encoding (preserves information)

\- No feature scaling (tree-based algorithm)

\- ~29 features (compact representation)

\- Native handling of high-cardinality features



Pipeline B: Logistic Regression \& MLP (Traditional)

\- One-hot encoding for categorical features

\- StandardScaler normalization

\- Binary encoding for some features

\- ~100+ features after encoding



Why This Matters:

\- CatBoost achieves +7.6% AUC over MLP by using optimized preprocessing

\- Demonstrates understanding of algorithm-specific requirements

\- Production-ready implementation with proper separation of concerns



 Model Training Strategy:



1\. Data Split: 70% train / 15% validation / 15% test

2\. Class Imbalance: Handled with SMOTE on training set

3\. Validation: Early stopping based on validation AUC for MLP

4\. Evaluation: Multiple metrics (AUC, Precision, Recall, F1)



---



 Tech Stack



Machine Learning:

\- CatBoost 1.2.2 (Gradient Boosting)

\- scikit-learn 1.3.2 (Logistic Regression, preprocessing)

\- PyTorch 2.1.2 (Neural Network)

\- imbalanced-learn 0.11.0 (SMOTE)



API \& Deployment:

\- FastAPI 0.109.0 (REST API)

\- Pydantic 2.5.3 (Data validation)

\- Uvicorn 0.27.0 (ASGI server)



Data Processing:

\- pandas 2.1.4

\- NumPy 1.26.2



Visualization:

\- Matplotlib 3.8.2

\- Seaborn 0.13.0



---



 Dataset



Source: Kaggle Vehicle Insurance Fraud Detection Dataset



Statistics:

\- Total records: 15,420

\- Features: 33

\- Target: Binary (Fraud / No Fraud)

\- Feature types: Numerical (8), Categorical (25)



Key Features:

\- Policy details (type, base policy, deductible)

\- Claim information (fault, police report, witnesses)

\- Vehicle data (make, category, age, price)

\- Personal information (age, sex, marital status)

\- Historical data (past claims, policy tenure)



---



 Running the Notebooks



Execute notebooks in order:



1\. 01\_eda.ipynb - Data exploration and visualization

2\. 02\_feature\_engineering.ipynb - Create dual preprocessing pipelines

3\. 03\_model\_training.ipynb - Train and compare models



Each notebook is self-contained with detailed explanations and visualizations.



---



 Future Improvements



\- Implement model explainability (SHAP values)

\- Add A/B testing framework for model comparison

\- Deploy to cloud (AWS/GCP)

\- Add monitoring and logging

\- Implement batch prediction endpoint

\- Create CI/CD pipeline

\- Add model retraining automation



---



 License



This project is licensed under the MIT License.


---



\ Screenshots



 API Documentation


<img width="1896" height="865" alt="Screenshot 2026-01-18 000628" src="https://github.com/user-attachments/assets/d26d7f9c-c9f6-4703-9fcf-9d6085eaf76a" />



 Prediction Example


<img width="1898" height="867" alt="Screenshot 2026-01-18 014344" src="https://github.com/user-attachments/assets/4d3312de-585b-4592-9ada-11d104267746" />



 Model Comparison


<img width="1688" height="598" alt="Screenshot 2026-01-20 054236" src="https://github.com/user-attachments/assets/f146d981-9b95-434f-9709-e7f168d623b9" />


---


If you found this project useful, please consider giving it a star!

