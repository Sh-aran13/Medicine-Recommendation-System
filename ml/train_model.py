from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from data_utils import load_severity_map, load_training_data, validate_cross_file_labels


ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT / "datasets"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model_bundle.joblib"


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    label_report = validate_cross_file_labels(DATASET_DIR)
    if label_report:
        print("WARNING: Label mismatch found across files:")
        for filename, labels in label_report.items():
            print(f"- {filename}: {len(labels)} unmatched labels")

    X, y, feature_columns = load_training_data(DATASET_DIR)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    top3 = np.mean([
        y_true in np.argsort(prob_row)[-3:]
        for y_true, prob_row in zip(y_test, y_prob)
    ])

    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Top-3 Accuracy: {top3:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    bundle = {
        "model": model,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "severity_map": load_severity_map(DATASET_DIR),
        "version": "v1-random-forest",
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved model bundle to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
