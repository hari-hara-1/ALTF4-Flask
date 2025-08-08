import streamlit as st
from explanation import explain_credit_score
from main import predict_credit_score  # Import from your main script

st.title("💳 Loan Credit Score & Decision Predictor")

# Collect user input
user_input = {}
user_input["age"] = st.number_input("Age", min_value=18, max_value=80, value=30)
user_input["num_occupants"] = st.number_input("Number of Occupants", min_value=1, max_value=10, value=2)
user_input["cash_inflow"] = st.number_input("Monthly Cash Inflow", min_value=0, value=50000)
user_input["avg_bank_balance"] = st.number_input("Average Bank Balance", min_value=0, value=10000)
user_input["bill_payment_consistency"] = st.selectbox("Bill Payment Consistency", [1.0, 0.7, 0.4, 0.2, 0.0])
user_input["bnpl_used"] = st.selectbox("BNPL Used?", [True, False])
user_input["bnpl ratio"] = st.slider("BNPL Ratio", 0.0, 1.0, 0.3)
user_input["rent_amount"] = st.number_input("Rent Amount", min_value=0, value=10000)
user_input["location_type"] = st.selectbox("Location Type", ["Urban", "Semi-Urban", "Rural"])
user_input["education_level"] = st.selectbox("Education Level", ["12th", "Diploma", "Graduate", "PostGraduate", "PhD"])
user_input["income_type"] = st.selectbox("Income Type", ["Salaried", "Gig", "Informal"])
user_input["grade_or_cgpa"] = st.slider("Grade or CGPA", 0.0, 10.0, 7.5)
user_input["housing_type"] = st.selectbox("Housing Type", ["Owned", "Rented", "Pg"])
user_input["age_to_employment_ratio"] = st.slider("Age to Employment Ratio", 0.0, 1.0, 0.5)

if st.button("Predict Credit Score"):
    score, _ = predict_credit_score(user_input)
    
    st.subheader(f"Predicted Credit Score: {score}")
    if score >= 680:
        st.success("✅ Loan Approved")
    elif score >= 600:
        st.warning("🟡 Loan Under Review")
    else:
        st.error("❌ Loan Rejected")
    st.subheader(f"Feature Weights on Credit:")
    
    shap_fig = explain_credit_score(user_input, show_plot=False)
    st.pyplot(shap_fig)
