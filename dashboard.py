import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load trained model
model = pickle.load(open("xgboost_default_model.pkl", "rb"))

st.title("📊 Loan Default Prediction Dashboard")

# --- Prediction Form ---
st.header("Enter Loan Details")

amount = st.number_input("Amount Borrowed", min_value=500, max_value=50000, step=500)
term = st.selectbox("Loan Term (months)", [12, 24, 36, 60])
rate = st.number_input("Borrower Rate (%)", min_value=1.0, max_value=50.0, step=0.1)
installment = st.number_input("Installment", min_value=100, max_value=5000, step=50)
grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
balance = st.number_input("Principal Balance", min_value=0, max_value=50000, step=500)
paid_principal = st.number_input("Principal Paid", min_value=0, max_value=50000, step=500)
paid_interest = st.number_input("Interest Paid", min_value=0, max_value=20000, step=100)
late_fees = st.number_input("Late Fees Paid", min_value=0, max_value=5000, step=10)
days_due = st.number_input("Days Past Due", min_value=0, max_value=1000, step=1)

if st.button("Predict Default Risk"):
    data = pd.DataFrame({
        "amount_borrowed": [amount],
        "term": [term],
        "borrower_rate": [rate],
        "installment": [installment],
        "grade": [ord(grade) - ord("A")],  # encode grade
        "principal_balance": [balance],
        "principal_paid": [paid_principal],
        "interest_paid": [paid_interest],
        "late_fees_paid": [late_fees],
        "days_past_due": [days_due]
    })
    prob = model.predict_proba(data)[0][1]
    if prob > 0.5:
        st.error(f"⚠️ Loan is LIKELY to default (Risk Probability: {prob:.2%})")
    else:
        st.success(f"✅ Loan is NOT likely to default (Risk Probability: {prob:.2%})")

# --- EDA Section ---
st.header("📊 Loan Data Insights")

df = pd.read_csv("cleaned_loans.csv")

# 1. Default Distribution
counts = df['default_flag'].value_counts()
fig1, ax1 = plt.subplots()
counts.plot(kind='bar', color=['green', 'red'], ax=ax1)
ax1.set_xticks([0,1])
ax1.set_xticklabels(['No Default (0)','Default (1)'])
ax1.set_title("Default Distribution")
st.pyplot(fig1)

# 2. Loan Amount vs Default
fig2, ax2 = plt.subplots()
df.boxplot(column='amount_borrowed', by='default_flag', grid=False, ax=ax2)
ax2.set_title("Loan Amount vs Default Flag")
ax2.set_xlabel("Default Flag")
ax2.set_ylabel("Amount Borrowed")
st.pyplot(fig2)
