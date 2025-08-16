import streamlit as st
import pandas as pd
import pickle
from xgboost import XGBClassifier

# Load trained model
with open("xgboost_default_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Loan Default Prediction Dashboard", layout="wide")

st.title("📊 Loan Default Prediction Dashboard")
st.markdown("This dashboard predicts whether a loan will default using the trained XGBoost model.")

# Sidebar - User Input
st.sidebar.header("Enter Loan Details")

def user_input_features():
    amount_borrowed = st.sidebar.number_input("Amount Borrowed", min_value=500, max_value=50000, value=10000)
    term = st.sidebar.selectbox("Loan Term (months)", [12, 24, 36, 60])
    borrower_rate = st.sidebar.slider("Borrower Rate (%)", 5.0, 40.0, 15.0)
    installment = st.sidebar.number_input("Installment", min_value=100, max_value=2000, value=300)
    grade = st.sidebar.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    principal_balance = st.sidebar.number_input("Principal Balance", min_value=0, max_value=50000, value=5000)
    principal_paid = st.sidebar.number_input("Principal Paid", min_value=0, max_value=50000, value=2000)
    interest_paid = st.sidebar.number_input("Interest Paid", min_value=0, max_value=20000, value=500)
    late_fees_paid = st.sidebar.number_input("Late Fees Paid", min_value=0, max_value=1000, value=0)
    days_past_due = st.sidebar.number_input("Days Past Due", min_value=0, max_value=365, value=0)

    data = {
        "amount_borrowed": amount_borrowed,
        "term": term,
        "borrower_rate": borrower_rate,
        "installment": installment,
        "grade": ord(grade) - ord("A"),  # Convert A-G to 0-6
        "principal_balance": principal_balance,
        "principal_paid": principal_paid,
        "interest_paid": interest_paid,
        "late_fees_paid": late_fees_paid,
        "days_past_due": days_past_due,
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Prediction
if st.button("🔍 Predict Default"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Loan is likely to DEFAULT (Risk Probability: {probability:.2%})")
    else:
        st.success(f"✅ Loan is NOT likely to default (Risk Probability: {probability:.2%})")
