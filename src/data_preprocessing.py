import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)

    # Drop customerID because it's not useful for prediction
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric (some values are blank)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Convert target column to binary
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # One-hot encode categorical columns
    df = pd.get_dummies(df, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")

    # Save processed dataset
    os.makedirs(output_path, exist_ok=True)
    pd.DataFrame(X_train_scaled).to_csv(f"{output_path}/X_train.csv", index=False)
    pd.DataFrame(X_test_scaled).to_csv(f"{output_path}/X_test.csv", index=False)
    y_train.to_csv(f"{output_path}/y_train.csv", index=False)
    y_test.to_csv(f"{output_path}/y_test.csv", index=False)

    print("✅ Data Preprocessing Completed Successfully!")
    print("Scaler saved in models/scaler.pkl")
    print("Processed files saved in:", output_path)


if __name__ == "__main__":
    preprocess_data(
        input_path="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        output_path="data/processed"
    )
