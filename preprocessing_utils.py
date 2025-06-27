# preprocessing_utils.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# These are the columns we'll encode and preserve order for
CATEGORICAL_COLS = [
    'gender', 'drug', 'genomics', 'past_diseases',
    'reason_for_drug', 'allergies', 'addiction',
    'ayurvedic_medicine', 'hereditary_disease', 'age_group'
]

FEATURE_ORDER = [
    "age", "gender", "drug", "genomics", "past_diseases", "reason_for_drug",
    "drug_quantity", "allergies", "addiction", "ayurvedic_medicine",
    "hereditary_disease", "drug_duration", "age_group", "polypharmacy_flag"
]

# Main preprocessing function used during training
def preprocess_data(df):
    df = df.copy()

    # Map target column
    df['adr_severity'] = df['adr_severity'].map({'Low': 0, 'High': 1})

    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # Compute target and input features
    X = df.drop(columns=['adr_severity'])
    y = df['adr_severity']

    return X[FEATURE_ORDER], y, encoders

def preprocess_input(input_json, encoders):
    data = input_json.copy()

    for col in CATEGORICAL_COLS:
        if col in data:
            try:
                data[col] = encoders[col].transform([data[col]])[0]
            except ValueError:
                # If unseen label, fallback to first class (or a known default)
                default = encoders[col].classes_[0]
                data[col] = encoders[col].transform([default])[0]

    data['polypharmacy_flag'] = 1 if data.get('drug_quantity', 0) > 3 else 0
    input_df = pd.DataFrame([data])
    return input_df[FEATURE_ORDER]
