import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# File paths
MODEL_PATH = 'credit_model.pkl'
ENCODER_PATH = 'label_encoders.pkl'
DATA_PATH = 'credit_score_database.csv'  # <- Your dataset path

# Categorical columns to encode
CATEGORICAL_COLUMNS = [
    'bnpl_used', 'location_type', 'education_level',
    'income_type', 'housing_type', 'loan_decision'  # Included for compatibility
]

# Features to train on
FEATURE_COLUMNS = [
    'age', 'num_occupants', 'cash_inflow', 'avg_bank_balance',
    'bill_payment_consistency', 'bnpl_used', 'bnpl ratio', 'rent_amount',
    'location_type', 'education_level', 'income_type', 'grade_or_cgpa',
    'housing_type', 'age_to_employment_ratio'
]

TARGET_COLUMN = 'credit_score'

def train_model(df: pd.DataFrame):
    df = df.copy()

    print("📊 Initial dataset shape:", df.shape)

    # Drop rows without the target
    df = df[df[TARGET_COLUMN].notna()]
    print("✅ Rows with valid credit_score:", df.shape)

    if df.empty:
        raise ValueError("Training data has no valid credit_score values.")

    # Initialize encoders
    label_encoders = {}

    # Encode categorical columns
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype(str)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            print(f"⚠️ Warning: Column '{col}' not found in dataset.")

    # Ensure all required features exist
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            raise KeyError(f"Missing required feature column: {col}")

    # Split features and target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model and encoders
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoders, ENCODER_PATH)

    print(f"✅ Model trained and saved to: {os.path.abspath(MODEL_PATH)}")
    print(f"✅ Label encoders saved to: {os.path.abspath(ENCODER_PATH)}")

# Run directly
if __name__ == "__main__":
    try:
        df = pd.read_csv(DATA_PATH)
        train_model(df)
    except Exception as e:
        print(f"❌ Error during training: {e}")
