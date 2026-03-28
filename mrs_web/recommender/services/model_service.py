from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from .knowledge_base import load_knowledge


ROOT_DIR = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT_DIR / "artifacts" / "model_bundle.joblib"


class ModelUnavailableError(RuntimeError):
    pass


def _normalize_symptom(symptom: str) -> str:
    return symptom.strip()


def predict_disease(selected_symptoms: List[str]) -> Dict[str, object]:
    if not MODEL_PATH.exists():
        raise ModelUnavailableError(
            "Model artifact not found. Run 'python ml/train_model.py' first."
        )

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    label_encoder = bundle["label_encoder"]
    feature_columns = bundle["feature_columns"]

    vector = {feature: 0 for feature in feature_columns}
    known_symptoms = set(load_knowledge()["symptoms"])

    for symptom in selected_symptoms:
        symptom = _normalize_symptom(symptom)
        if symptom in known_symptoms:
            vector[symptom] = 1

    X = pd.DataFrame([vector], columns=feature_columns)

    pred_idx = model.predict(X)[0]
    disease = str(label_encoder.inverse_transform(np.array([pred_idx]))[0]).strip()

    probabilities = model.predict_proba(X)[0]
    sorted_idx = np.argsort(probabilities)[::-1][:3]

    top_candidates = [
        {
            "disease": str(label_encoder.inverse_transform(np.array([i]))[0]).strip(),
            "probability": float(probabilities[i]),
        }
        for i in sorted_idx
    ]

    return {
        "predicted_disease": disease,
        "confidence": float(max(probabilities)),
        "top_candidates": top_candidates,
    }
