from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


DATASET_DIR = Path(__file__).resolve().parents[1] / "datasets"


def normalize_text(value: str) -> str:
    """Normalize label text to reduce whitespace/casing mismatches."""
    if pd.isna(value):
        return ""
    return " ".join(str(value).replace("_", " ").strip().lower().split())


def load_training_data(dataset_dir: Path | None = None) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    base_dir = dataset_dir or DATASET_DIR
    train_path = base_dir / "Training.csv"
    df = pd.read_csv(train_path)

    # The training file already stores binary symptom columns and final label column.
    feature_columns = [c for c in df.columns if c != "prognosis"]
    X = df[feature_columns].copy()
    y = df["prognosis"].astype(str).str.strip()

    return X, y, feature_columns


def load_severity_map(dataset_dir: Path | None = None) -> Dict[str, int]:
    base_dir = dataset_dir or DATASET_DIR
    severity_path = base_dir / "Symptom-severity.csv"
    severity_df = pd.read_csv(severity_path)
    severity_df["Symptom"] = severity_df["Symptom"].astype(str).str.strip()
    return dict(zip(severity_df["Symptom"], severity_df["weight"]))


def validate_cross_file_labels(dataset_dir: Path | None = None) -> Dict[str, List[str]]:
    """Check whether disease labels are aligned across key files.

    Returns missing-label report keyed by filename.
    """
    base_dir = dataset_dir or DATASET_DIR

    training = pd.read_csv(base_dir / "Training.csv")
    disease_train = {normalize_text(v) for v in training["prognosis"].dropna().astype(str)}

    files_and_columns = [
        ("medications.csv", "Disease"),
        ("diets.csv", "Disease"),
        ("description.csv", "Disease"),
        ("precautions_df.csv", "Disease"),
    ]

    report: Dict[str, List[str]] = {}
    for filename, col in files_and_columns:
        df = pd.read_csv(base_dir / filename)
        disease_other = {normalize_text(v) for v in df[col].dropna().astype(str)}
        missing = sorted(disease_train - disease_other)
        if missing:
            report[filename] = missing

    return report
