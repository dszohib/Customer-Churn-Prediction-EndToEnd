import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_model():
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv")
    y_test = y_test.values.ravel()

    model = joblib.load("models/churn_model.pkl")

    predictions = model.predict(X_test)

    print("✅ Model Evaluation Results")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))

if __name__ == "__main__":
    evaluate_model()
