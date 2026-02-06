import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train_model():
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv")

    y_train = y_train.values.ravel()

    # Model 1: Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    # Model 2: Random Forest
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    # Compare training accuracy
    lr_acc = accuracy_score(y_train, lr.predict(X_train))
    rf_acc = accuracy_score(y_train, rf.predict(X_train))

    print(f"Logistic Regression Training Accuracy: {lr_acc:.4f}")
    print(f"Random Forest Training Accuracy: {rf_acc:.4f}")

    # Select best model
    best_model = rf if rf_acc > lr_acc else lr

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/churn_model.pkl")

    print("✅ Best Model Saved as models/churn_model.pkl")

if __name__ == "__main__":
    train_model()
