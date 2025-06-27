# train_model.py
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
from preprocessing_utils import preprocess_data

# 1. Load Dataset
def load_data(path='data/smart_adr_dataset.csv'):
    return pd.read_csv(path)

# 2. Train Model
def train_model(X_train, y_train):
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    return model

# 3. Save Model
def save_model(model, path='models/xgb_model.pkl'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

# 4. Save Encoders
def save_encoders(encoders, path='models/encoders.pkl'):
    joblib.dump(encoders, path)

# 5. Main Training Pipeline
def main():
    print("🔄 Loading data...")
    df = load_data()

    print("🧹 Preprocessing...")
    X, y, encoders = preprocess_data(df)

    print("🧠 Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("📈 Training model...")
    model = train_model(X_train, y_train)

    print("💾 Saving model and encoders...")
    save_model(model)
    save_encoders(encoders)

    print("✅ Evaluating model...")
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    main()
