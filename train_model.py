from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

DATA_FILE = Path("data/signs.csv")
MODEL_FILE = Path("models/sign_classifier.joblib")

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            "No training data found. Run collect_data.py for at least two signs first."
        )

    df = pd.read_csv(DATA_FILE)

    if df["label"].nunique() < 2:
        raise ValueError("You need at least TWO different sign labels before training.")

    X = df.drop(columns=["label"])
    y = df["label"]

    print("\nSamples per sign:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)

    print(f"\nValidation accuracy: {accuracy * 100:.2f}%")
    print("\nClassification report:")
    print(classification_report(y_test, pred, zero_division=0))

    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print(f"\nSaved trained model to: {MODEL_FILE}")

if __name__ == "__main__":
    main()
