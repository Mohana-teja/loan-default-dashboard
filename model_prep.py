import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("cleaned_loans.csv")

# Select features (you can expand later)
features = ["amount_borrowed", "term", "borrower_rate", "grade"]
target = "default_flag"

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df[features], drop_first=True)

# Split into train & test
X_train, X_test, y_train, y_test = train_test_split(
    df_encoded, df[target], test_size=0.2, random_state=42, stratify=df[target]
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
