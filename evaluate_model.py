import pandas as pd
import joblib
import matplotlib.pyplot as plt

def evaluate_credit_score(input_data: dict, model_path="credit_score_model.pkl"):
    model = joblib.load(model_path)
    input_df = pd.DataFrame([input_data])
    prob = model.predict_proba(input_df)[0, 1]
    score = int(300 + (850 - 300) * prob)

    print(f"\n🧾 Estimated Credit Score (scale 300–850): {score}")
    print(f"📈 Loan Approval Probability: {prob:.2f}")

    # Feature contributions
    transformed_input = model.named_steps["preprocessor"].transform(input_df)
    weights = model.named_steps["classifier"].coef_[0]
    contributions = transformed_input[0] * weights

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    feature_contribs = pd.Series(contributions, index=feature_names).sort_values()

    print("\n🔍 Key Negative Influences:")
    for feature, value in feature_contribs.items():
        if value < 0:
            print(f"❌ {feature}: {value:.2f}")

    feature_contribs.plot(kind="barh", figsize=(10, 6), title="Feature Contributions")
    plt.axvline(0, color='black', linestyle='--')
    plt.tight_layout()
    plt.show()

    return score
