
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from preprocessing_utils import preprocess_input


model = joblib.load("models/xgb_model.pkl")
encoders = joblib.load("models/encoders.pkl")

app = FastAPI(title="ADR Inference API")

class PatientData(BaseModel):
    age: int
    gender: str
    drug: str
    genomics: str
    past_diseases: str
    reason_for_drug: str
    drug_quantity: int
    allergies: str
    addiction: str
    ayurvedic_medicine: str
    hereditary_disease: str
    drug_duration: int
    age_group: str

@app.get("/")
def read_root():
    return {"message": "ADR Inference API is running."}

@app.post("/predict")
def predict_adr_severity(data: PatientData):
    try:
        
        input_dict = data.dict()
        processed_df = preprocess_input(input_dict, encoders)

        prediction = model.predict(processed_df)[0]
        proba = float(model.predict_proba(processed_df)[0][1])

        label = "High" if prediction == 1 else "Low"

        return {
            "adr_severity": label,
            "confidence": round(proba if label == "High" else 1 - proba, 3)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
