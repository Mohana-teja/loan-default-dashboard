import pandas as pd

# Load dataset
df = pd.read_csv('Master_Loan_Summary.csv')

# Drop listing_title column
df = df.drop(columns=['listing_title'])

print(df.head())
# Create target column: default_flag
df['default_flag'] = df['loan_status_description'].apply(
    lambda x: 1 if x in ['CHARGED OFF', 'DEFAULTED'] else 0
)

print(df[['loan_status_description', 'default_flag']].head(10))
# Keep only useful columns
df = df[['amount_borrowed', 'term', 'borrower_rate', 'installment',
         'grade', 'principal_balance', 'principal_paid', 'interest_paid',
         'late_fees_paid', 'days_past_due', 'default_flag']]

print(df.head())
# Save the cleaned dataset
df.to_csv('cleaned_loans.csv', index=False)

print(" Cleaned dataset saved as 'cleaned_loans.csv'")
