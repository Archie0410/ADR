import streamlit as st
import requests
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import StringIO

st.set_page_config(page_title="ADR Severity Predictor", page_icon="💊", layout="wide")
st.title("💊 ADR Severity Risk Predictor")
st.markdown("Enter patient data to get a real-time prediction of ADR (Adverse Drug Reaction) severity.")

model = joblib.load("models/xgb_model.pkl")
explainer = shap.Explainer(model)
encoders = joblib.load("models/encoders.pkl")


with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=0, max_value=120, value=60)
        gender = st.selectbox("Gender", encoders["gender"].classes_)
        drug = st.selectbox("Drug", encoders["drug"].classes_)
        genomics = st.selectbox("Genomic Marker", encoders["genomics"].classes_)
        past_diseases = st.selectbox("Past Disease History", encoders["past_diseases"].classes_)
        reason_for_drug = st.selectbox("Reason for Drug", encoders["reason_for_drug"].classes_)
        drug_quantity = st.number_input("Number of Drugs Prescribed", min_value=1, max_value=10, value=2)

    with col2:
        allergies = st.selectbox("Allergies", encoders["allergies"].classes_)
        addiction = st.selectbox("Addiction", encoders["addiction"].classes_)
        ayurvedic_medicine = st.selectbox("Ayurvedic Medicine Use", encoders["ayurvedic_medicine"].classes_)
        hereditary_disease = st.selectbox("Hereditary Disease", encoders["hereditary_disease"].classes_)
        drug_duration = st.number_input("Drug Duration (days)", min_value=1, max_value=365, value=30)
        age_group = st.selectbox("Age Group", encoders["age_group"].classes_)
        threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)

    submitted = st.form_submit_button("🧠 Predict ADR Severity")


def check_drug_gene_rules(drug, genomics, past_diseases):
    warnings = []
    if drug.lower() == "aspirin" and genomics.lower() == "geneb":
        warnings.append("⚠ Aspirin with GeneB increases ADR risk.")
    if drug.lower() == "amoxicillin" and "kidney" in past_diseases.lower():
        warnings.append("⚠ Amoxicillin not recommended for Kidney Disease.")
    return warnings

if submitted:
    input_payload = {
        "age": age,
        "gender": gender,
        "drug": drug,
        "genomics": genomics,
        "past_diseases": past_diseases,
        "reason_for_drug": reason_for_drug,
        "drug_quantity": drug_quantity,
        "allergies": allergies,
        "addiction": addiction,
        "ayurvedic_medicine": ayurvedic_medicine,
        "hereditary_disease": hereditary_disease,
        "drug_duration": drug_duration,
        "age_group": age_group
    }

    with st.spinner("Sending data to inference API..."):
        try:
            response = requests.post("http://localhost:8000/predict", json=input_payload)
            if response.status_code == 200:
                result = response.json()
                prob = result['confidence']

                colA, colB = st.columns([1, 1.2])

                with colA:
                    if prob >= threshold:
                        st.success(f"🔬 Predicted ADR Severity: **{result['adr_severity']}** (Confidence: {prob * 100:.1f}%)")
                    else:
                        st.info(f"Below confidence threshold ({threshold}), result: {result['adr_severity']}")

                    st.markdown("### 🔍 Feature Impact (SHAP Explanation)")
                    try:
                        def encode_input_for_shap(input_dict, encoders):
                            data = input_dict.copy()
                            encoded = {}
                            for k, v in data.items():
                                if k in encoders:
                                    encoded[k] = encoders[k].transform([v])[0]
                                else:
                                    encoded[k] = v
                            # Add polypharmacy_flag for SHAP input shape
                            encoded["polypharmacy_flag"] = 1 if encoded["drug_quantity"] > 3 else 0
                            ordered_cols = [
                                "age", "gender", "drug", "genomics", "past_diseases", "reason_for_drug",
                                "drug_quantity", "allergies", "addiction", "ayurvedic_medicine",
                                "hereditary_disease", "drug_duration", "age_group", "polypharmacy_flag"
                            ]
                            return pd.DataFrame([encoded])[ordered_cols]

                        shap_input_df = encode_input_for_shap(input_payload, encoders)
                        shap_values = explainer(shap_input_df)
                        fig, ax = plt.subplots()
                        shap.plots.bar(shap_values[0], show=False)
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"SHAP Explanation unavailable: {e}")

                    warnings = check_drug_gene_rules(drug, genomics, past_diseases)
                    for warn in warnings:
                        st.error(warn)

                with colB:
                    st.markdown("### 📊 ADR Risk Meter")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=prob * 100,
                        title={"text": "ADR Risk (%)"},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "crimson" if prob > 0.5 else "green"},
                            "steps": [
                                {"range": [0, 50], "color": "#d4edda"},
                                {"range": [50, 100], "color": "#f8d7da"}
                            ]
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error(f"API Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")
