import shap
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import numpy as np

MODEL_PATH = "credit_model.pkl"
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

# Load assets
model = joblib.load(MODEL_PATH)
label_encoders = joblib.load(ENCODER_PATH)
background_df = pd.read_csv(BACKGROUND_DATA_PATH)

def preprocess_input(user_input):
    df = pd.DataFrame([user_input])
    for col in label_encoders:
        if col in df.columns:
            df[col] = df[col].apply(str)
            known_classes = set(label_encoders[col].classes_)
            df[col] = df[col].apply(lambda x: x if x in known_classes else label_encoders[col].classes_[0])
            df[col] = label_encoders[col].transform(df[col])
    return df[FEATURE_ORDER].astype(float)

def explain_credit_score(user_input, show_plot=True):
    processed = preprocess_input(user_input)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(processed)
    base_value = float(np.array(explainer.expected_value).flatten()[0])

    print(f"\n📊 Base Score (average prediction): {round(base_value, 2)}")

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[0],
            base_values=base_value,
            data=processed.iloc[0],
            feature_names=processed.columns
        ),
        max_display=10
    )

    if show_plot:
        plt.show()
    return fig