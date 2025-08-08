import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from update_model import retrain_model_with_input

import warnings
warnings.filterwarnings("ignore")

# Constants
MODEL_PATH = "credit_model.pkl"  # Consistent with training
ENCODER_PATH = "label_encoders.pkl"
BACKGROUND_DATA_PATH = "credit_score_database.csv"

FEATURE_ORDER = [
    "age",
    "num_occupants",
    "cash_inflow",
    "avg_bank_balance",
    "bill_payment_consistency",
    "bnpl_used",
    "bnpl ratio",
    "rent_amount",
    "location_type",
    "education_level",
    "income_type",
    "grade_or_cgpa",
    "housing_type",
    "age_to_employment_ratio"
]

# Load model and encoders
model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODER_PATH)
background_df = pd.read_csv(BACKGROUND_DATA_PATH)

# --- PREPROCESS FUNCTIONS ---

def preprocess_dataframe(df):
    df = df.copy()

    # Keep only columns needed
    df = df[[col for col in FEATURE_ORDER if col in df.columns]]

    for col in label_encoders:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, bool) else x)
            df[col] = df[col].fillna(np.nan)

            known_classes = set(label_encoders[col].classes_)
            df[col] = df[col].apply(lambda x: x if x in known_classes else label_encoders[col].classes_[0])

            df[col] = label_encoders[col].transform(df[col])

    df = df[FEATURE_ORDER]
    return df.astype(float)

def preprocess_input(user_input):
    input_df = pd.DataFrame([user_input])
    for col in label_encoders:
        if col in input_df.columns:
            input_df[col] = input_df[col].apply(lambda x: str(x) if isinstance(x, bool) else x)
            input_df[col] = input_df[col].fillna(np.nan)

            known_classes = set(label_encoders[col].classes_)
            input_df[col] = input_df[col].apply(lambda x: x if x in known_classes else label_encoders[col].classes_[0])

            input_df[col] = label_encoders[col].transform(input_df[col])
    input_df = input_df[FEATURE_ORDER]
    return input_df.astype(float)

# --- SHAP SETUP ---

background_X = preprocess_dataframe(background_df)
background_X_sample = background_X.sample(n=100, random_state=42)

# --- PREDICTION + EXPLANATION ---

def predict_credit_score(user_input):
    processed = preprocess_input(user_input)
    prediction = model.predict(processed)
    return round(prediction[0], 2), processed

def explain_prediction(processed_input):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(processed_input)

    print("\n📉 SHAP Feature Impact (Negative means lowering score):")
    for name, value in zip(processed_input.columns, shap_values[0]):
        print(f"{name}: {value:.2f}")

    shap.plots.waterfall(shap.Explanation(values=shap_values[0],
                                          base_values=explainer.expected_value,
                                          data=processed_input.iloc[0],
                                          feature_names=processed_input.columns), max_display=10)

# --- SAMPLE INPUTS ---

sample_users = [
    {
        "age": 25,
        "num_occupants": 2,
        "cash_inflow": 45000,
        "avg_bank_balance": 8000,
        "bill_payment_consistency": 1.0,
        "bnpl_used": False,
        "bnpl ratio": 0.0,
        "rent_amount": 8000,
        "location_type": "Urban",
        "education_level": "Graduate",
        "income_type": "Salaried",
        "grade_or_cgpa": 7.8,
        "housing_type": "Rented",
        "age_to_employment_ratio": 0.65,
        "expected_credit_score": 680  # High due to good banking + no BNPL + strong payment
    },
    {
        "age": 35,
        "num_occupants": 4,
        "cash_inflow": 30000,
        "avg_bank_balance": 1500,
        "bill_payment_consistency": 0.4,
        "bnpl_used": True,
        "bnpl ratio": 0.8,
        "rent_amount": 12000,
        "location_type": "Rural",
        "education_level": "12th",
        "income_type": "Informal",
        "grade_or_cgpa": 6.0,
        "housing_type": "Pg",
        "age_to_employment_ratio": 0.4,
        "expected_credit_score": 490  # Low due to bad BNPL, low consistency, and rent stress
    },
    {
        "age": 29,
        "num_occupants": 1,
        "cash_inflow": 75000,
        "avg_bank_balance": 15000,
        "bill_payment_consistency": 0.7,
        "bnpl_used": True,
        "bnpl ratio": 0.3,
        "rent_amount": 10000,
        "location_type": "Semi-Urban",
        "education_level": "PostGraduate",
        "income_type": "Salaried",
        "grade_or_cgpa": 8.5,
        "housing_type": "Owned",
        "age_to_employment_ratio": 0.7,
        "expected_credit_score": 720  # Excellent profile
    },
    {
        "age": 41,
        "num_occupants": 5,
        "cash_inflow": 25000,
        "avg_bank_balance": 2000,
        "bill_payment_consistency": 0.2,
        "bnpl_used": True,
        "bnpl ratio": 0.9,
        "rent_amount": 15000,
        "location_type": "Rural",
        "education_level": "Diploma",
        "income_type": "Gig",
        "grade_or_cgpa": 5.5,
        "housing_type": "Rented",
        "age_to_employment_ratio": 0.3,
        "expected_credit_score": 450  # Poor financial profile
    },
    {
        "age": 33,
        "num_occupants": 3,
        "cash_inflow": 55000,
        "avg_bank_balance": 5000,
        "bill_payment_consistency": 0.7,
        "bnpl_used": False,
        "bnpl ratio": 0.0,
        "rent_amount": 9000,
        "location_type": "Urban",
        "education_level": "Graduate",
        "income_type": "Salaried",
        "grade_or_cgpa": 7.0,
        "housing_type": "Owned",
        "age_to_employment_ratio": 0.6,
        "expected_credit_score": 660  # Well-balanced profile
    }
]


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    score, processed = predict_credit_score(sample_users[4])
    print(f"\n📊 Predicted Credit Score: {score}")
    explain_prediction(processed)

    # Optional: retrain model with new input
    retrain_model_with_input(sample_users[4])