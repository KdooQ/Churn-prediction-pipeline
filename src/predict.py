import joblib
import pandas as pd

from src.data import load_data


def predict_churn(customers: pd.DataFrame, model_path="models/churn_model.joblib"):
    model = joblib.load(model_path)
    return model.predict_proba(customers)[:, 1]


if __name__ == "__main__":
    df = load_data()
    sample = df.drop(columns="Churn").head(5)
    probs = predict_churn(sample)
    for p, real in zip(probs, df["Churn"].head(5)):
        print(f"Probabilidad de abandono: {p:.2%} (real: {real})")