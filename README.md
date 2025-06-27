
# 💊 ADR Prediction Model

A machine learning project that predicts the risk of Adverse Drug Reactions (ADRs) based on patient Electronic Health Records (EHR). 

## 🚀 Project Structure

```
ADR_Model_Project/
├── data/
│   └── ehr.csv                # Sample EHR data
├── models/
│   ├── xgb_model.pkl          # Trained XGBoost model
│   └── encoders.pkl           # Saved label encoders
├── app/
│   └── streamlit_app.py       # Streamlit UI
├── main.py                    # Core model training pipeline
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 📦 Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/your-username/ADR_Model_Project.git
cd ADR_Model_Project
```

2. **Create & activate virtual environment (optional but recommended)**
```bash
python -m venv venv
# Activate:
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install requirements**
```bash
python -m pip install -r requirements.txt
```

## 🧠 Train the Model
```bash
python main.py
```
This will:
- Load the data from `data/ehr.csv`
- Train an XGBoost model
- Save the model and encoders to `models/`

## 🌐 Run the Streamlit App
```bash
python -m streamlit run app/streamlit_app.py
```
You’ll get a form to:
- Enter age, gender, drug, genomics marker
- Get instant ADR risk prediction

## 📊 Sample Features

- `age`: Integer (0–100)
- `gender`: Male / Female
- `drug`: e.g., Paracetamol, Ibuprofen
- `genomics`: e.g., GeneA, GeneB
- `adr_label`: 0 (No ADR) or 1 (ADR)

## ✅ Future Enhancements (Optional Ideas)
- Use real-world clinical data (after de-identification)
- Add SHAP explainability for model transparency
- Connect to a live EHR system via API
- Replace Streamlit with Gradio or Flask

## 👩‍⚕️ Disclaimer

This is a simulated research project and **not intended for real medical decision-making**.
