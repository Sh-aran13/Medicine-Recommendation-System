from __future__ import annotations

import ast
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[3]
DATASET_DIR = ROOT_DIR / "datasets"


def _clean_text(value: str) -> str:
    return " ".join(str(value).strip().split())


@lru_cache(maxsize=1)
def load_knowledge() -> Dict[str, object]:
    training = pd.read_csv(DATASET_DIR / "Training.csv")
    medications = pd.read_csv(DATASET_DIR / "medications.csv")
    descriptions = pd.read_csv(DATASET_DIR / "description.csv")
    precautions = pd.read_csv(DATASET_DIR / "precautions_df.csv")
    diets = pd.read_csv(DATASET_DIR / "diets.csv")
    workouts = pd.read_csv(DATASET_DIR / "workout_df.csv")

    symptoms = [c for c in training.columns if c != "prognosis"]

    med_map: Dict[str, List[str]] = {}
    for _, row in medications.iterrows():
        disease = _clean_text(row["Disease"])
        raw = row["Medication"]
        try:
            meds = ast.literal_eval(raw)
            meds = [_clean_text(m) for m in meds]
        except (ValueError, SyntaxError):
            meds = [_clean_text(raw)]
        med_map[disease] = meds

    desc_map = {
        _clean_text(r["Disease"]): _clean_text(r["Description"])
        for _, r in descriptions.iterrows()
    }

    precaution_map: Dict[str, List[str]] = {}
    for _, r in precautions.iterrows():
        disease = _clean_text(r["Disease"])
        values = [
            _clean_text(r.get("Precaution_1", "")),
            _clean_text(r.get("Precaution_2", "")),
            _clean_text(r.get("Precaution_3", "")),
            _clean_text(r.get("Precaution_4", "")),
        ]
        precaution_map[disease] = [v for v in values if v and v != "nan"]

    diet_map: Dict[str, List[str]] = {}
    for _, row in diets.iterrows():
        disease = _clean_text(row["Disease"])
        raw = row["Diet"]
        try:
            values = ast.literal_eval(raw)
            values = [_clean_text(v) for v in values]
        except (ValueError, SyntaxError):
            values = [_clean_text(raw)]
        diet_map[disease] = values

    workout_map: Dict[str, List[str]] = {}
    for _, row in workouts.iterrows():
        disease = _clean_text(row["disease"])
        workout = _clean_text(row["workout"])
        workout_map.setdefault(disease, []).append(workout)

    return {
        "symptoms": symptoms,
        "medications": med_map,
        "descriptions": desc_map,
        "precautions": precaution_map,
        "diets": diet_map,
        "workouts": workout_map,
    }
