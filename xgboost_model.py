import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib   # for saving model

# Load train dataset
df = pd.read_csv("cleaned_loans.csv")
print("Columns in dataset:", df.columns)

# Encode categorical columns (like 'grade')
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Features and target
X = df.drop("default_flag", axis=1)
y = df["default_flag"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train XGBoost
print("Training XGBoost model...")
model = XGBClassifier(eval_metric="logloss")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Results
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\n✅ XGBoost training completed.")

# Save trained model
joblib.dump(model, "xgboost_default_model.pkl")
print("💾 Model saved as xgboost_default_model.pkl")
