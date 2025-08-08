import pandas as pd
import joblib
import os
from trainmodel import train_model
from sklearn.exceptions import NotFittedError

# Constants
DATA_PATH = "credit_score_database.csv"
MODEL_PATH = "credit_model.pkl"
ENCODER_PATH = "label_encoders.pkl"

# Features used in model
FEATURE_COLUMNS = [
    'age', 'num_occupants', 'cash_inflow', 'avg_bank_balance',
    'bill_payment_consistency', 'bnpl_used', 'bnpl ratio', 'rent_amount',
    'location_type', 'education_level', 'income_type', 'grade_or_cgpa',
    'housing_type', 'age_to_employment_ratio'
]

# --- Retraining Function ---

def retrain_model_with_input(input_data: dict):
    # Load existing data
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Database not found at: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Load model and encoders
    try:
        model = joblib.load(MODEL_PATH)
        label_encoders = joblib.load(ENCODER_PATH)
    except (FileNotFoundError, NotFittedError):
        raise RuntimeError("Model or encoders not found. Please train the model first.")

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Handle categorical columns
    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)
            known_classes = set(le.classes_)
            input_df[col] = input_df[col].apply(lambda x: x if x in known_classes else le.classes_[0])
            input_df[col] = le.transform(input_df[col])
        else:
            input_df[col] = 0  # default if missing

    # Reorder & fill missing features
    for col in FEATURE_COLUMNS:
        if col not in input_df.columns:
            input_df[col] = df[col].mode()[0] if col in df.columns else 0

    input_df = input_df[FEATURE_COLUMNS].astype(float)

    # Predict credit score
    predicted_score = model.predict(input_df)[0]
    print(f"📊 Predicted score added to new data: {predicted_score:.2f}")

    # Add score to original input
    input_data['credit_score'] = round(predicted_score, 2)

    # Append to database
    new_row = pd.DataFrame([input_data])
    updated_df = pd.concat([df, new_row], ignore_index=True)
    updated_df.to_csv(DATA_PATH, index=False)
    print("📁 New data added to credit_score_database.csv")

    # Retrain model on updated data
    train_model(updated_df)
