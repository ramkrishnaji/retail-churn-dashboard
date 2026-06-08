"""
Step 4: FastAPI Deployment
Author: Student
Course: Retail Analytics Project

This script deploys our retail analytics models as a REST API using FastAPI.
Available Endpoints:
1. POST /predict:
   - Takes raw customer inputs (Recency, Frequency, Monetary, AOV, Review Score, Payment Method, Cancellation Rate).
   - Standardizes RFM features.
   - Predicts customer segment using the saved K-Means model.
   - Formats the features and feeds them to the XGBoost model to get churn probability and binary prediction.
   - Returns both the segment label and churn predictions.
2. GET /segments:
   - Computes and returns centroid statistics and sizes of each customer segment from our dataset.

To run this app locally:
uvicorn 4_api:app --host 127.0.0.1 --port 8000 --reload
"""

import os
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Create FastAPI app instance
app = FastAPI(
    title="Retail Customer Segmentation & Churn API",
    description="FastAPI service for predicting customer churn and classifying customer segments.",
    version="1.0"
)

# Define file paths for models and artifacts
SCALER_PATH = "models/scaler.pkl"
KMEANS_PATH = "models/kmeans.pkl"
MAPPING_PATH = "models/cluster_mapping.pkl"
XGB_PATH = "models/xgboost_model.pkl"
ENCODER_PATH = "models/encoder.pkl"
FEATURES_PATH = "models/feature_list.pkl"
DATA_PATH = "data/processed/segmented_customers.csv.gz"

# Global variables for loaded models
scaler = None
kmeans = None
cluster_mapping = None
xgb_model = None
encoder = None
feature_list = None

@app.on_event("startup")
def load_models():
    """Load all serialized machine learning models and encoders at server startup."""
    global scaler, kmeans, cluster_mapping, xgb_model, encoder, feature_list
    
    try:
        print("Loading models from models/ directory...")
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        with open(KMEANS_PATH, "rb") as f:
            kmeans = pickle.load(f)
        with open(MAPPING_PATH, "rb") as f:
            cluster_mapping = pickle.load(f)
        with open(XGB_PATH, "rb") as f:
            xgb_model = pickle.load(f)
        with open(ENCODER_PATH, "rb") as f:
            encoder = pickle.load(f)
        with open(FEATURES_PATH, "rb") as f:
            feature_list = pickle.load(f)
        print("All models loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please ensure you have run 1_data_prep.py, 2_segmentation.py, and 3_churn_prediction.py first.")

# ----------------------------------------------------
# Pydantic Schemas for Input/Output Validation
# ----------------------------------------------------
class CustomerInput(BaseModel):
    recency: float = Field(..., description="Days since last purchase (relative to snapshot date)", example=45.0)
    frequency: int = Field(..., description="Total number of orders", example=3)
    monetary: float = Field(..., description="Total monetary spend", example=250.50)
    avg_order_value: float = Field(..., description="Average value per order", example=83.50)
    avg_review_score: float = Field(..., description="Average review score given by customer (1-5)", example=4.3)
    preferred_payment_method: str = Field(..., description="Payment method used most often", example="credit_card")
    cancellation_rate: float = Field(..., description="Proportion of orders canceled (0 to 1)", example=0.0)

class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="Predicted probability of churning")
    churn_prediction: int = Field(..., description="Binary churn prediction (0 = Active, 1 = Churned)")
    segment_label: str = Field(..., description="Customer segment label determined by K-Means")

# ----------------------------------------------------
# Endpoints
# ----------------------------------------------------
@app.get("/")
def read_root():
    return {"message": "Retail Customer Segmentation & Churn API is active. Go to /docs for Swagger UI."}

@app.post("/predict", response_model=PredictionResponse)
def predict_churn_and_segment(customer: CustomerInput):
    """
    Predict the segment and churn probability for an individual customer.
    """
    # Verify that models are loaded
    if None in [scaler, kmeans, cluster_mapping, xgb_model, encoder, feature_list]:
        raise HTTPException(status_code=500, detail="Models are not loaded on server startup.")
        
    try:
        # Step 1: Predict customer cluster segment
        # Extract RFM features, scale them, and predict
        rfm_data = np.array([[customer.recency, customer.frequency, customer.monetary]])
        rfm_scaled = scaler.transform(rfm_data)
        cluster_id = kmeans.predict(rfm_scaled)[0]
        segment_label = cluster_mapping[cluster_id]
        
        # Step 2: Prepare features for the supervised churn model
        # Create a single row DataFrame for the input customer features (excluding recency to avoid leakage)
        numeric_features = {
            "frequency": customer.frequency,
            "monetary": customer.monetary,
            "avg_order_value": customer.avg_order_value,
            "avg_review_score": customer.avg_review_score,
            "cancellation_rate": customer.cancellation_rate
        }
        
        categorical_features = {
            "preferred_payment_method": customer.preferred_payment_method,
            "segment": segment_label
        }
        
        # Encode categorical variables using the pre-fitted OneHotEncoder
        cat_df = pd.DataFrame([categorical_features])
        cat_encoded = encoder.transform(cat_df)
        cat_cols = encoder.get_feature_names_out(list(categorical_features.keys()))
        encoded_df = pd.DataFrame(cat_encoded, columns=cat_cols)
        
        # Merge numeric and encoded categorical features
        num_df = pd.DataFrame([numeric_features])
        input_features = pd.concat([num_df, encoded_df], axis=1)
        
        # Align features with the exact sequence used during model training
        # Ensure any missing columns (e.g. unseen payment method) are filled with 0
        input_final = pd.DataFrame(columns=feature_list)
        for col in feature_list:
            if col in input_features.columns:
                input_final.at[0, col] = input_features.at[0, col]
            else:
                input_final.at[0, col] = 0.0
                
        input_final = input_final.astype(float)
        
        # Step 3: Run Churn Prediction
        churn_prob = float(xgb_model.predict_proba(input_final)[0, 1])
        churn_pred = int(xgb_model.predict(input_final)[0])
        
        return PredictionResponse(
            churn_probability=churn_prob,
            churn_prediction=churn_pred,
            segment_label=segment_label
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/segments")
def get_segment_statistics():
    """
    Get summary statistics and sizing details for all customer segments.
    """
    if not os.path.exists(DATA_PATH):
        raise HTTPException(
            status_code=404, 
            detail="Segmented customer dataset not found. Run customer segmentation first."
        )
        
    try:
        # Load the segmented customer features
        segmented_df = pd.read_csv(DATA_PATH)
        
        # Calculate summary statistics per customer segment
        summary = segmented_df.groupby('segment').agg(
            customer_count=('customer_unique_id', 'count'),
            avg_recency=('recency', 'mean'),
            avg_frequency=('frequency', 'mean'),
            avg_monetary=('monetary', 'mean'),
            avg_review_score=('avg_review_score', 'mean'),
            avg_cancellation_rate=('cancellation_rate', 'mean')
        ).reset_index()
        
        # Add descriptions for a student-style explanation of the clusters
        descriptions = {
            "Champions": "High spenders, buying frequently, and purchased recently. Highly active and loyal.",
            "Loyal Customers": "Regular buyers with decent spend and frequency, showing active engagement.",
            "New Customers": "Recently made their first purchase. Low frequency and monetary spend, but high potential.",
            "At-Risk Customers": "Historically good customers who have not made a purchase in a long time. High churn risk.",
            "Lost Customers": "One-time or low-value buyers who purchased a long time ago and have disengaged."
        }
        
        summary['description'] = summary['segment'].map(descriptions)
        
        # Format values to 2 decimal places for cleaner API response
        for col in ['avg_recency', 'avg_frequency', 'avg_monetary', 'avg_review_score', 'avg_cancellation_rate']:
            summary[col] = summary[col].round(2)
            
        return summary.to_dict(orient='records')
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate segment stats: {str(e)}")
