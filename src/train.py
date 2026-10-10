import joblib
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

from src.data import load_data
from src.features import build_preprocessor


def main():
    df = load_data()
    X = df.drop(columns="Churn")
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    models = {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=42
        ),
    }

    results = {}
    for name, model in models.items():
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("model", model)])
        pipe.fit(X_train, y_train)
        auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        results[name] = (auc, pipe)
        print(f"\n{name}: ROC-AUC = {auc:.3f}")
        print(classification_report(y_test, pipe.predict(X_test)))

    best = max(results, key=lambda n: results[n][0])
    joblib.dump(results[best][1], "models/churn_model.joblib")
    print(f"Modelo guardado: {best}")


if __name__ == "__main__":
    main()