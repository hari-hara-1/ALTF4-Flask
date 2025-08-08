def evaluate_credit_score(user_input: dict, model, min_score=300, max_score=850, return_suggestions=False):
    import pandas as pd
    import numpy as np

    input_df = pd.DataFrame([user_input])
    prob = model.predict_proba(input_df)[0, 1]
    credit_score = int(min_score + (max_score - min_score) * prob)

    # Transform input
    transformed_input = model.named_steps["preprocessor"].transform(input_df)
    weights = model.named_steps["classifier"].coef_[0]
    contributions = transformed_input[0] * weights

    # Get feature names
    preprocessor = model.named_steps["preprocessor"]
    num_features = preprocessor.transformers_[0][2]
    cat_features = preprocessor.transformers_[1][2]
    ohe = preprocessor.named_transformers_["cat"]
    cat_ohe_features = ohe.get_feature_names_out(cat_features)
    all_feature_names = list(num_features) + list(cat_ohe_features)

    # Contributions
    import pandas as pd
    feature_contribs = pd.Series(contributions, index=all_feature_names).sort_values()

    feature_advice = {
        # ... same advice dict as before ...
        "cash_inflow": "Your cash inflow is low — increasing your income could improve your score.",
        "avg_bank_balance": "Maintain a higher average bank balance for better creditworthiness.",
        "rent_amount": "Reduce your rent burden if possible — high rent affects your score negatively.",
        "num_occupants": "Consider reducing financial dependency (number of household members).",
        "grade_or_cgpa": "Improving your academic performance may help in long-term credit evaluation.",
        "age_to_employment_ratio": "Try to increase your work experience relative to your age.",
        "income_type_Informal": "Shifting to a more stable income source (like salaried work) could help.",
        "income_type_Gig": "Gig work is seen as less stable — salaried roles may improve approval chances.",
        "bill_payment_consistency_Sometimes": "Try to consistently pay your bills on time.",
        "bill_payment_consistency_Rarely": "Irregular bill payments are a major red flag. Improve this urgently.",
        "bnpl_used_False": "Using BNPL responsibly can help build credit history.",
        "housing_type_Rented": "Owning a home can be a positive financial indicator.",
        "education_level_12th": "Higher education levels are linked with better approval odds.",
        "education_level_Diploma": "Pursuing higher education may positively impact your score."
    }

    negative = feature_contribs[feature_contribs < 0]
    suggestions = []

    for feature in negative.index:
        for key in feature_advice:
            if key in feature:
                suggestions.append(feature_advice[key])
                break

    if not suggestions:
        suggestions.append("No major issues found — your profile is solid!")

    if return_suggestions:
        return credit_score, suggestions

    # Else: full CLI/plot experience
    print(f"\n🧾 Credit Score (300–850): {credit_score}")
    print("\n🔻 Suggestions to Improve Your Credit Score:")
    for s in suggestions:
        print(f"❌ {s}")

    # Plot (optional)
    import matplotlib.pyplot as plt
    feature_contribs.plot(kind='barh', figsize=(10, 7), title="Feature Contributions to Credit Score")
    plt.axvline(0, color='black', linestyle='--')
    plt.tight_layout()
    plt.show()

    return credit_score
