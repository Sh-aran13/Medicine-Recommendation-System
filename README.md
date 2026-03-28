# Medicine Recommendation System (MRS)

AI-assisted symptom-to-disease prediction and recommendation system built with:

- Python + scikit-learn (Random Forest model)
- Django web application (form input + result page)
- Local CSV knowledge base for medications, precautions, diet, and workouts

The project predicts a likely disease from selected symptoms and returns supportive recommendations.

## 1. What This Project Does

Given user inputs:

- age
- gender
- symptoms (multi-select)

The system:

- builds a symptom vector compatible with the trained model
- predicts disease using RandomForestClassifier
- returns confidence and top-3 candidate diseases
- enriches output using local knowledge data:
- disease description
- medication suggestions
- precautions
- diet suggestions
- workout/lifestyle suggestions

## 2. Current System Behavior

- Main input page: Django form template
- Submit action: standard POST to backend route
- Result page: separate styled HTML template rendered by Django
- API mode: if request header `X-Requested-With: XMLHttpRequest` is sent, backend returns JSON instead of full HTML

## 3. Tech Stack

- Python 3.10+
- Django 5.x
- pandas, numpy
- scikit-learn
- joblib
- fpdf2 (report generation)
- python-docx (installed on-demand by research script)

## 4. Project Structure

```text
MRS/
├─ datasets/
│  ├─ Training.csv
│  ├─ symptoms_df.csv
│  ├─ medications.csv
│  ├─ description.csv
│  ├─ precautions_df.csv
│  ├─ Symptom-severity.csv
│  ├─ diets.csv
│  └─ workout_df.csv
├─ artifacts/
│  └─ model_bundle.joblib
├─ ml/
│  ├─ data_utils.py
│  └─ train_model.py
├─ mrs_web/
│  ├─ manage.py
│  ├─ db.sqlite3
│  ├─ mrs_web/
│  │  ├─ settings.py
│  │  └─ urls.py
│  └─ recommender/
│     ├─ forms.py
│     ├─ views.py
│     ├─ urls.py
│     ├─ services/
│     │  ├─ knowledge_base.py
│     │  ├─ model_service.py
│     │  └─ rule_engine.py
│     └─ templates/recommender/
│        ├─ form.html
│        └─ result.html
├─ generate_research_documents.py
├─ requirements.txt
└─ README.md
```

## 5. Data Files and Roles

The project uses local CSV files in `datasets/`.

1. `Training.csv`
- Binary symptom feature columns + `prognosis` label.
- Source for model training and disease classes.
- Columns:
```text
itching, skin_rash, nodal_skin_eruptions, continuous_sneezing, shivering, chills, joint_pain, stomach_pain, acidity, ulcers_on_tongue, muscle_wasting, vomiting, burning_micturition, spotting_ urination, fatigue, weight_gain, anxiety, cold_hands_and_feets, mood_swings, weight_loss, restlessness, lethargy, patches_in_throat, irregular_sugar_level, cough, high_fever, sunken_eyes, breathlessness, sweating, dehydration, indigestion, headache, yellowish_skin, dark_urine, nausea, loss_of_appetite, pain_behind_the_eyes, back_pain, constipation, abdominal_pain, diarrhoea, mild_fever, yellow_urine, yellowing_of_eyes, acute_liver_failure, fluid_overload, swelling_of_stomach, swelled_lymph_nodes, malaise, blurred_and_distorted_vision, phlegm, throat_irritation, redness_of_eyes, sinus_pressure, runny_nose, congestion, chest_pain, weakness_in_limbs, fast_heart_rate, pain_during_bowel_movements, pain_in_anal_region, bloody_stool, irritation_in_anus, neck_pain, dizziness, cramps, bruising, obesity, swollen_legs, swollen_blood_vessels, puffy_face_and_eyes, enlarged_thyroid, brittle_nails, swollen_extremeties, excessive_hunger, extra_marital_contacts, drying_and_tingling_lips, slurred_speech, knee_pain, hip_joint_pain, muscle_weakness, stiff_neck, swelling_joints, movement_stiffness, spinning_movements, loss_of_balance, unsteadiness, weakness_of_one_body_side, loss_of_smell, bladder_discomfort, foul_smell_of urine, continuous_feel_of_urine, passage_of_gases, internal_itching, toxic_look_(typhos), depression, irritability, muscle_pain, altered_sensorium, red_spots_over_body, belly_pain, abnormal_menstruation, dischromic _patches, watering_from_eyes, increased_appetite, polyuria, family_history, mucoid_sputum, rusty_sputum, lack_of_concentration, visual_disturbances, receiving_blood_transfusion, receiving_unsterile_injections, coma, stomach_bleeding, distention_of_abdomen, history_of_alcohol_consumption, fluid_overload, blood_in_sputum, prominent_veins_on_calf, palpitations, painful_walking, pus_filled_pimples, blackheads, scurring, skin_peeling, silver_like_dusting, small_dents_in_nails, inflammatory_nails, blister, red_sore_around_nose, yellow_crust_ooze, prognosis
```

2. `Symptom-severity.csv`
- Maps symptom to severity weight.
- Included in saved model bundle for downstream use.
- Columns:
```text
Symptom, weight
```

3. `medications.csv`
- Columns include `Disease`, `Medication`.
- `Medication` typically stores list-like strings.
- Columns:
```text
Disease, Medication
```

4. `description.csv`
- Disease text descriptions.
- Columns:
```text
Disease, Description
```

5. `precautions_df.csv`
- Precaution columns (`Precaution_1` to `Precaution_4`).
- Columns:
```text
Unnamed: 0, Disease, Precaution_1, Precaution_2, Precaution_3, Precaution_4
```

6. `diets.csv`
- Disease-to-diet mapping (list-like strings).
- Columns:
```text
Disease, Diet
```

7. `workout_df.csv`
- Disease-to-workout/lifestyle guidance.
- Columns:
```text
Unnamed: 0, disease, workout
```

8. `symptoms_df.csv`
- Additional symptom listing used as reference.
- Columns:
```text
Unnamed: 0, Disease, Symptom_1, Symptom_2, Symptom_3, Symptom_4
```

## 6. Training Pipeline

Training script: `ml/train_model.py`

Training steps:

1. Loads and validates cross-file disease label consistency.
2. Loads features and labels from `Training.csv`.
3. Encodes labels with `LabelEncoder`.
4. Splits data with stratified 80/20 train/test.
5. Trains `RandomForestClassifier`:
- `n_estimators=300`
- `class_weight="balanced_subsample"`
- `random_state=42`
- `n_jobs=-1`
6. Prints metrics:
- Macro F1
- Top-3 Accuracy
- Classification report
7. Saves bundle to `artifacts/model_bundle.joblib`.

Bundle contents:

- `model`
- `label_encoder`
- `feature_columns`
- `severity_map`
- `version`

## 7. Web Application Flow

Backend form schema (`recommender/forms.py`):

- `age`: integer, range 0 to 120
- `gender`: one of `male`, `female`, `other`
- `symptoms`: multiple-choice, populated dynamically from knowledge base

Routes:

- `/` -> input form page
- `/recommend/` -> prediction endpoint

Prediction flow in `recommender/views.py`:

1. Build symptom choices from knowledge base.
2. Validate submitted form.
3. Run model prediction (`model_service.predict_disease`).
4. Merge with knowledge maps (description, meds, precautions, diet, workout).
5. Return either:
- HTML result page (normal form post)
- JSON (`{"ok": true, "result": ...}`) for AJAX requests

## 8. Recommendation and Rules

Knowledge loading:

- `services/knowledge_base.py` reads all CSV files and builds in-memory maps.
- Uses `@lru_cache(maxsize=1)` to avoid repeated dataset reads.

Rule engine:

- `services/rule_engine.py` contains example medication safety checks:
- allergy blocking
- simple contraindication rules
- basic interaction checks

Important:

- The current web response path primarily returns direct dataset recommendations.
- `rule_engine.py` is present for extension and safety logic evolution.

## 9. Installation and Setup

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Windows (CMD)

```bat
python -m venv .venv
.venv\Scripts\activate.bat
pip install --upgrade pip
pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 10. Run Training

From project root:

```bash
python ml/train_model.py
```

Expected output:

- metrics printed in terminal
- model bundle created/updated at `artifacts/model_bundle.joblib`

## 11. Run the Django App

```bash
cd mrs_web
python manage.py migrate
python manage.py runserver
```

Open: `http://127.0.0.1:8000/`


## 12. API Behavior (Optional JSON Mode)

Endpoint: `POST /recommend/`

If header is present:

`X-Requested-With: XMLHttpRequest`

Success response format:

```json
{
	"ok": true,
	"result": {
		"disease": "...",
		"confidence": 0.99,
		"top_candidates": [
			{"disease": "...", "probability": 0.99},
			{"disease": "...", "probability": 0.01}
		],
		"description": "...",
		"medications": ["..."],
		"precautions": ["..."],
		"diet": ["..."],
		"workout": ["..."],
		"medical_disclaimer": "..."
	}
}
```

Validation error response:

```json
{
	"ok": false,
	"errors": {
		"field_name": [
			{"message": "...", "code": "..."}
		]
	}
}
```

## 13. Configuration Notes

Current project settings (`mrs_web/mrs_web/settings.py`):

- `DEBUG = True`
- `ALLOWED_HOSTS = ["*"]`
- SQLite database at `mrs_web/db.sqlite3`

Before production deployment:

1. Set a secure `SECRET_KEY`.
2. Set `DEBUG = False`.
3. Restrict `ALLOWED_HOSTS`.
4. Configure static files, HTTPS, and production server.

## 14. Troubleshooting

1. Error: model artifact not found
- Cause: training not run yet.
- Fix: run `python ml/train_model.py` from project root.

2. Form loads but prediction fails
- Cause: missing dependencies or corrupted artifact.
- Fix: reinstall requirements and retrain model.

3. Dataset mismatch warnings during training
- Cause: disease labels differ across CSVs.
- Fix: normalize disease names across all dataset files.

4. Template changes not visible
- Cause: browser cache.
- Fix: hard refresh (`Ctrl+F5`) and restart dev server.

5. CSRF or POST issues
- Cause: missing token or manual API call mismatch.
- Fix: include CSRF token and proper headers for AJAX.

## 15. Limitations

- Dataset-driven and educational, not clinically validated.
- Model quality depends fully on training data quality.
- Current features do not include lab values or medical imaging.
- Recommendation rules are limited and should be clinically reviewed before real-world use.

## 16. Safety Disclaimer

This application is a decision-support learning project only.
It does not provide medical diagnosis or treatment.
Always consult a licensed medical professional for clinical decisions.

## 17. Requirements

Current pinned dependency ranges (`requirements.txt`):

- Django>=5.0,<6.0
- pandas>=2.2.0
- numpy>=1.26.0
- scikit-learn>=1.4.0
- joblib>=1.3.0
- fpdf2>=2.7.0

## 18. Quick Start (Shortest Path)

```bash
pip install -r requirements.txt
python ml/train_model.py
cd mrs_web
python manage.py migrate
python manage.py runserver
```

Then open `http://127.0.0.1:8000/` and run a prediction.
