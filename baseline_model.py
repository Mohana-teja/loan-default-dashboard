import pandas as pd
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=100, solver='saga', n_jobs=-1)
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("cleaned_loans.csv")

# Select features
features = ["amount_borrowed", "term", "borrower_rate", "grade"]
target = "default_flag"

# One-hot encode categorical variables
df_encoded = pd.get_dummies(df[features], drop_first=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    df_encoded, df[target], test_size=0.2, random_state=42, stratify=df[target]
)

# Train Logistic Regression
model = LogisticRegression(max_iter=1000, class_weight="balanced")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print(" Baseline model script started...")
