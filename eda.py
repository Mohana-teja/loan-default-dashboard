import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load trained model
model = joblib.load("xgboost_default_model.pkl")

st.title("📊 Loan Default Prediction Dashboard")

# ---- Prediction Form ----
st.header("Enter Loan Details")

amount_borrowed = st.number_input("Amount Borrowed", min_value=0)
term = st.number_input("Loan Term (months)", min_value=0)
borrower_rate = st.number_input("Borrower Rate (%)", min_value=0.0)
installment = st.number_input("Installment", min_value=0.0)
grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
principal_balance = st.number_input("Principal Balance", min_value=0.0)
principal_paid = st.number_input("Principal Paid", min_value=0.0)
interest_paid = st.number_input("Interest Paid", min_value=0.0)
late_fees_paid = st.number_input("Late Fees Paid", min_value=0.0)
days_past_due = st.number_input("Days Past Due", min_value=0)

# Map grade to numeric
grade_map = {"A":0, "B":1, "C":2, "D":3, "E":4, "F":5, "G":6}

if st.button("Predict Loan Default"):
    input_data = pd.DataFrame([[
        amount_borrowed, term, borrower_rate, installment,
        grade_map[grade], principal_balance, principal_paid,
        interest_paid, late_fees_paid, days_past_due
    ]], columns=[
        "amount_borrowed", "term", "borrower_rate", "installment",
        "grade", "principal_balance", "principal_paid",
        "interest_paid", "late_fees_paid", "days_past_due"
    ])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1] * 100

    if prediction == 1:
        st.error(f"⚠️ Loan is LIKELY to default (Risk Probability: {probability:.2f}%)")
    else:
        st.success(f"✅ Loan is NOT likely to default (Risk Probability: {probability:.2f}%)")

# ---- EDA Section ----
st.header("📊 Loan Data Insights")

df = pd.read_csv("cleaned_loans.csv")

# 1. Default Distribution
fig1, ax1 = plt.subplots()
df['default_flag'].value_counts().plot(kind='bar', color=['green','red'], ax=ax1)
ax1.set_xticklabels(['No Default (0)','Default (1)'], rotation=0)
ax1.set_title("Default Distribution")
st.pyplot(fig1)

# 2. Loan Amount vs Default Flag
fig2, ax2 = plt.subplots()
df.boxplot(column='amount_borrowed', by='default_flag', grid=False, ax=ax2)
ax2.set_title("Loan Amount vs Default Flag")
ax2.set_xlabel("Default Flag (0 = No, 1 = Yes)")
ax2.set_ylabel("Amount Borrowed")
st.pyplot(fig2)

# 3. Interest Rate vs Default Flag
fig3, ax3 = plt.subplots()
df.boxplot(column='borrower_rate', by='default_flag', grid=False, ax=ax3)
ax3.set_title("Interest Rate vs Default Flag")
ax3.set_xlabel("Default Flag (0 = No, 1 = Yes)")
ax3.set_ylabel("Borrower Rate")
st.pyplot(fig3)

# 4. Default Rate by Loan Grade
fig4, ax4 = plt.subplots()
df.groupby("grade")["default_flag"].mean().sort_values().plot(kind="bar", ax=ax4)
ax4.set_title("Default Rate by Loan Grade")
ax4.set_xlabel("Loan Grade")
ax4.set_ylabel("Default Rate")
st.pyplot(fig4)

# 5. Default Rate by Loan Term
fig5, ax5 = plt.subplots()
df.groupby("term")["default_flag"].mean().plot(kind="bar", ax=ax5)
ax5.set_title("Default Rate by Loan Term")
ax5.set_xlabel("Loan Term (Months)")
ax5.set_ylabel("Default Rate")
st.pyplot(fig5)
