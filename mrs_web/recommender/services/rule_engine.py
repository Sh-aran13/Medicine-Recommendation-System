from __future__ import annotations

from typing import Dict, List


# Extend this table with clinically reviewed interactions.
DRUG_INTERACTIONS = {
    ("Antihistamines", "Epinephrine"): "caution",
    ("Antibiotics", "Antifungal Cream"): "safe",
    ("Metformin", "Insulin"): "caution",
}

# Example contraindication table. Replace with medically validated rules before production.
CONTRAINDICATION_RULES = {
    "pregnancy": {"Methotrexate", "Ergotamine derivatives"},
    "liver_disease": {"Ketoconazole"},
    "kidney_disease": {"Metformin"},
}


def _tokenize_csv_text(text: str) -> List[str]:
    if not text:
        return []
    return [t.strip() for t in text.split(",") if t.strip()]


def evaluate_medications(
    meds: List[str],
    age: int,
    gender: str,
    medical_history_text: str,
    allergies_text: str,
    current_medications_text: str,
) -> Dict[str, List[Dict[str, object]]]:
    allergies = {a.lower() for a in _tokenize_csv_text(allergies_text)}
    history = {h.lower() for h in _tokenize_csv_text(medical_history_text)}
    current = _tokenize_csv_text(current_medications_text)

    accepted = []
    blocked = []
    cautions = []

    for med in meds:
        med_key = med.lower()
        reasons = []
        blocked_flag = False

        if med_key in allergies:
            blocked_flag = True
            reasons.append("Blocked: listed in user allergies.")

        if age < 12 and "corticosteroids" in med_key:
            reasons.append("Caution: pediatric use requires physician supervision.")

        if gender.lower() == "female" and "pregnancy" in history:
            if med in CONTRAINDICATION_RULES.get("pregnancy", set()):
                blocked_flag = True
                reasons.append("Blocked: contraindicated in pregnancy.")

        for history_key in ("liver_disease", "kidney_disease"):
            if history_key in history and med in CONTRAINDICATION_RULES.get(history_key, set()):
                blocked_flag = True
                reasons.append(f"Blocked: contraindicated in {history_key.replace('_', ' ')}.")

        for current_med in current:
            pair = (med, current_med)
            reverse_pair = (current_med, med)
            interaction = DRUG_INTERACTIONS.get(pair) or DRUG_INTERACTIONS.get(reverse_pair)
            if interaction == "caution":
                reasons.append(f"Caution: possible interaction with {current_med}.")
            elif interaction == "block":
                blocked_flag = True
                reasons.append(f"Blocked: severe interaction with {current_med}.")

        item = {"medication": med, "reasons": reasons}

        if blocked_flag:
            blocked.append(item)
        elif reasons:
            cautions.append(item)
            accepted.append(item)
        else:
            accepted.append(item)

    return {
        "accepted": accepted,
        "blocked": blocked,
        "cautions": cautions,
    }
