import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# Load trained model
# ------------------------------
with open("xgboost_default_model.pkl", "rb") as f:
    model = pickle.load(f)

# Page setup
st.set_page_config(page_title="Loan Default Prediction Dashboard", layout="wide")
st.title(" Loan Default Prediction Dashboard")
st.markdown("This dashboard predicts whether a loan will default using the trained XGBoost model and also shows loan data insights.")

# ------------------------------
# Sidebar - User Input
# ------------------------------
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

# ------------------------------
# Prediction Section
# ------------------------------
st.subheader("🔍 Prediction Result")
if st.button("Predict Default"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Loan is LIKELY to DEFAULT (Risk Probability: {probability:.2%})")
    else:
        st.success(f"✅ Loan is NOT likely to default (Risk Probability: {probability:.2%})")

# ------------------------------
# Loan Data Insights (EDA + Analytics)
# ------------------------------
st.header(" Loan Data Insights")

# Load small dataset (for deployment)
df = pd.read_csv("cleaned_loans_small.csv")

# ---- Basic Analytics ----
st.subheader("Dataset Overview")
st.write(df.head())
st.write(f"**Number of rows:** {df.shape[0]}")
st.write(f"**Number of columns:** {df.shape[1]}")
st.write(f"**Missing values:** {df.isnull().sum().sum()}")

# ---- Plots ----
st.subheader("EDA Visualizations")

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

st.success(" Prediction + 5 Plots + Analytics loaded successfully!")
