import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import shap
import matplotlib.pyplot as plt


model = joblib.load("models/xgb_model.pkl")
encoders = joblib.load("models/encoders.pkl")

st.set_page_config(page_title="ADR Severity Predictor", page_icon="💊", layout="wide")
st.title("💊 ADR Severity Risk Predictor")

col1, col2 = st.columns([1, 1])


with col1:
    st.header("📝 Patient Details")

    age = st.number_input("Age", min_value=0, max_value=120, value=60)
    gender = st.selectbox("Gender", encoders["gender"].classes_)
    drug = st.selectbox("Drug", encoders["drug"].classes_)
    genomics = st.selectbox("Genomic Marker", encoders["genomics"].classes_)
    past_diseases = st.selectbox("Past Disease History", encoders["past_diseases"].classes_)
    reason_for_drug = st.selectbox("Reason for Drug", encoders["reason_for_drug"].classes_)
    drug_quantity = st.number_input("Number of Drugs Prescribed", min_value=1, max_value=20, value=2)
    allergies = st.selectbox("Allergies", encoders["allergies"].classes_)
    addiction = st.selectbox("Addiction", encoders["addiction"].classes_)
    ayurvedic = st.selectbox("Ayurvedic Medicine Use", encoders["ayurvedic_medicine"].classes_)
    hereditary = st.selectbox("Hereditary Disease", encoders["hereditary_disease"].classes_)
    drug_duration = st.slider("Drug Duration (days)", 1, 180, 30)
    age_group = st.selectbox("Age Group", encoders["age_group"].classes_)

    polypharmacy_flag = 1 if drug_quantity > 3 else 0

    def encode(col, val):
        return encoders[col].transform([val])[0]

    input_data = {
        "age": age,
        "gender": encode("gender", gender),
        "drug": encode("drug", drug),
        "genomics": encode("genomics", genomics),
        "past_diseases": encode("past_diseases", past_diseases),
        "reason_for_drug": encode("reason_for_drug", reason_for_drug),
        "drug_quantity": drug_quantity,
        "allergies": encode("allergies", allergies),
        "addiction": encode("addiction", addiction),
        "ayurvedic_medicine": encode("ayurvedic_medicine", ayurvedic),
        "hereditary_disease": encode("hereditary_disease", hereditary),
        "drug_duration": drug_duration,
        "age_group": encode("age_group", age_group),
        "polypharmacy_flag": polypharmacy_flag
    }

    input_df = pd.DataFrame([input_data])
    expected_order = [
        "age", "gender", "drug", "genomics", "past_diseases", "reason_for_drug",
        "drug_quantity", "allergies", "addiction", "ayurvedic_medicine",
        "hereditary_disease", "drug_duration", "age_group", "polypharmacy_flag"
    ]
    input_df = input_df[expected_order]

with col2:
    st.header("🔍 Prediction Result")

    if st.button("🧠 Predict ADR Severity"):
        prediction_encoded = model.predict(input_df)[0]
        prediction_label = encoders["adr_severity"].inverse_transform([prediction_encoded])[0]
        prob_high = model.predict_proba(input_df)[0][1]

        if prediction_label == "High":
            st.error(f"⚠ **High ADR Risk** (Confidence: `{prob_high:.2f}`)")
        else:
            st.success(f"✅ **Low ADR Risk** (Confidence: `{1 - prob_high:.2f}`)")

       
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_high * 100,
            title={"text": "ADR Risk (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red" if prob_high >= 0.5 else "green"},
                "steps": [
                    {"range": [0, 50], "color": "#d4edda"},
                    {"range": [50, 100], "color": "#f8d7da"},
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

      
        st.subheader("📊 SHAP Feature Contribution")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_df)

        excluded_features = ["polypharmacy_flag", "age_group"]
        included_features = [col for col in input_df.columns if col not in excluded_features]

        shap_values_df = pd.DataFrame(shap_values, columns=input_df.columns)
        shap_values_filtered = shap_values_df[included_features]

        fig_shap, ax = plt.subplots()
        shap.summary_plot(shap_values_filtered.values, input_df[included_features], plot_type="bar", show=False)
        st.pyplot(fig_shap)
