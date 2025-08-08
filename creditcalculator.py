def calculate_credit_score(data):
    # Weights for each category
    weights = {
        "income_to_rent": 150,
        "bill_payment": 150,
        "bnpl_usage": 100,
        "bank_balance_ratio": 100,
        "education": 50,
        "age_employment_ratio": 50,
        "location_housing": 50,
        "num_occupants": 50,
        "income_type": 100
    }

    score = 0

    # 1. Income-to-Rent Ratio
    income_to_rent_ratio = data["cash_inflow"] / data["rent_amount"]
    if income_to_rent_ratio >= 4:
        score += weights["income_to_rent"]
    elif income_to_rent_ratio >= 2:
        score += weights["income_to_rent"] * 0.75
    else:
        score += weights["income_to_rent"] * 0.5

    # 2. Bill Payment Consistency
    bill_map = {"Always": 1.0, "Usually": 0.7, "Sometimes": 0.4, "Rarely": 0.2, "Never": 0.0}
    bill_score = bill_map.get(data["bill_payment_consistency"], 0.5)
    score += weights["bill_payment"] * bill_score

    # 3. BNPL Usage
    if data["bnpl_used"]:
        if data["bnpl ratio"] < 0.5:
            score += weights["bnpl_usage"] * 0.6
        elif data["bnpl ratio"] < 0.8:
            score += weights["bnpl_usage"] * 0.4
        else:
            score += weights["bnpl_usage"] * 0.2
    else:
        score += weights["bnpl_usage"]

    # 4. Bank Balance-to-Income Ratio
    bank_ratio = data["avg_bank_balance"] / data["cash_inflow"]
    if bank_ratio >= 0.1:
        score += weights["bank_balance_ratio"]
    elif bank_ratio >= 0.05:
        score += weights["bank_balance_ratio"] * 0.7
    else:
        score += weights["bank_balance_ratio"] * 0.4

    # 5. Education Level & CGPA
    if data["education_level"] in ["Postgraduate", "Graduate"]:
        if data["grade_or_cgpa"] >= 8:
            score += weights["education"]
        elif data["grade_or_cgpa"] >= 6:
            score += weights["education"] * 0.75
        else:
            score += weights["education"] * 0.5
    else:
        score += weights["education"] * 0.5

    # 6. Age-to-Employment Ratio
    if data["age_to_employment_ratio"] >= 0.6:
        score += weights["age_employment_ratio"]
    elif data["age_to_employment_ratio"] >= 0.4:
        score += weights["age_employment_ratio"] * 0.7
    else:
        score += weights["age_employment_ratio"] * 0.4

    # 7. Location & Housing
    if data["location_type"] == "Urban" and data["housing_type"] == "Owned":
        score += weights["location_housing"]
    elif data["location_type"] == "Urban" or data["housing_type"] == "Owned":
        score += weights["location_housing"] * 0.7
    else:
        score += weights["location_housing"] * 0.4

    # 8. Number of Occupants
    if data["num_occupants"] <= 2:
        score += weights["num_occupants"]
    elif data["num_occupants"] <= 4:
        score += weights["num_occupants"] * 0.75
    else:
        score += weights["num_occupants"] * 0.5

    # 9. Income Type
    if data["income_type"] == "Salaried":
        score += weights["income_type"]
    else:
        score += weights["income_type"] * 0.6

    return round(score, 2)
