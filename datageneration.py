import random
import pandas as pd

def generate_synthetic_user():
    return {
        "age": random.randint(21, 60),
        "num_occupants": random.randint(1, 6),
        "cash_inflow": random.randint(15000, 100000),
        "avg_bank_balance": random.randint(500, 30000),
        "bill_payment_consistency": random.choices(
            [1.0, 0.7, 0.4, 0.2, 0.0],  # Already numeric!
            [0.4, 0.3, 0.15, 0.1, 0.05]
        )[0],
        "bnpl_used": random.choice([True, False]),
        "bnpl ratio": round(random.uniform(0, 1), 2),
        "rent_amount": random.randint(5000, 40000),
        "location_type": random.choice(["Urban", "Semi-Urban", "Rural"]),
        "education_level": random.choice(["12th", "Diploma", "Graduate", "PostGraduate", "PhD"]),
        "income_type": random.choice(["Salaried", "Gig", "Informal"]),
        "grade_or_cgpa": round(random.uniform(5.0, 10.0), 1),
        "housing_type": random.choice(["Owned", "Rented", "Pg"]),
        "age_to_employment_ratio": round(random.uniform(0.2, 0.9), 2)
    }

def calculate_credit_score(data):
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

    income_to_rent_ratio = data["cash_inflow"] / max(data["rent_amount"], 1)
    if income_to_rent_ratio >= 4:
        score += weights["income_to_rent"]
    elif income_to_rent_ratio >= 2:
        score += weights["income_to_rent"] * 0.75
    else:
        score += weights["income_to_rent"] * 0.5

    score += weights["bill_payment"] * data["bill_payment_consistency"]

    if data["bnpl_used"]:
        if data["bnpl ratio"] < 0.5:
            score += weights["bnpl_usage"] * 0.6
        elif data["bnpl ratio"] < 0.8:
            score += weights["bnpl_usage"] * 0.4
        else:
            score += weights["bnpl_usage"] * 0.2
    else:
        score += weights["bnpl_usage"]

    bank_ratio = data["avg_bank_balance"] / max(data["cash_inflow"], 1)
    if bank_ratio >= 0.1:
        score += weights["bank_balance_ratio"]
    elif bank_ratio >= 0.05:
        score += weights["bank_balance_ratio"] * 0.7
    else:
        score += weights["bank_balance_ratio"] * 0.4

    if data["education_level"] in ["PostGraduate", "PhD"]:
        if data["grade_or_cgpa"] >= 8:
            score += weights["education"]
        elif data["grade_or_cgpa"] >= 6:
            score += weights["education"] * 0.75
        else:
            score += weights["education"] * 0.5
    elif data["education_level"] in ["Graduate", "Diploma"]:
        score += weights["education"] * 0.75
    else:
        score += weights["education"] * 0.5

    if data["age_to_employment_ratio"] >= 0.6:
        score += weights["age_employment_ratio"]
    elif data["age_to_employment_ratio"] >= 0.4:
        score += weights["age_employment_ratio"] * 0.7
    else:
        score += weights["age_employment_ratio"] * 0.4

    if data["location_type"] == "Urban" and data["housing_type"] == "Owned":
        score += weights["location_housing"]
    elif data["location_type"] == "Urban" or data["housing_type"] == "Owned":
        score += weights["location_housing"] * 0.7
    else:
        score += weights["location_housing"] * 0.4

    if data["num_occupants"] <= 2:
        score += weights["num_occupants"]
    elif data["num_occupants"] <= 4:
        score += weights["num_occupants"] * 0.75
    else:
        score += weights["num_occupants"] * 0.5

    if data["income_type"] == "Salaried":
        score += weights["income_type"]
    else:
        score += weights["income_type"] * 0.6

    return round(score, 2)

def loan_decision(score):
    if score >= 700:
        return "Approved"
    elif score >= 600:
        return "Review"
    else:
        return "Rejected"

def generate_user_database(n=1000):
    users = []
    for _ in range(n):
        user = generate_synthetic_user()
        user["credit_score"] = calculate_credit_score(user)
        user["loan_decision"] = loan_decision(user["credit_score"])
        users.append(user)
    return pd.DataFrame(users)

# Generate and save dataset
df = generate_user_database(1000)
df.to_csv("credit_score_database.csv", index=False)
print("✅ Dataset saved as credit_score_database.csv")
