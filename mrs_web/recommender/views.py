from __future__ import annotations

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST

from .forms import RecommendationForm
from .services.knowledge_base import load_knowledge
from .services.model_service import ModelUnavailableError, predict_disease


def _build_form() -> RecommendationForm:
    kb = load_knowledge()
    form = RecommendationForm()
    sorted_symptoms = sorted(
        kb["symptoms"],
        key=lambda s: s.replace("_", " ").strip().lower(),
    )
    symptom_choices = [(s, s.replace("_", " ").title()) for s in sorted_symptoms]
    form.fields["symptoms"].choices = symptom_choices
    return form


@require_GET
def index(request):
    form = _build_form()
    return render(request, "recommender/form.html", {"form": form})


@require_POST
def recommend(request):
    kb = load_knowledge()
    form = _build_form()

    submitted = RecommendationForm(request.POST)
    submitted.fields["symptoms"].choices = form.fields["symptoms"].choices

    if not submitted.is_valid():
        errors = submitted.errors.get_json_data()
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse({"ok": False, "errors": errors}, status=400)
        return render(request, "recommender/form.html", {"form": submitted, "errors": errors})

    data = submitted.cleaned_data

    try:
        pred = predict_disease(data["symptoms"])
    except ModelUnavailableError as exc:
        payload = {
            "ok": False,
            "errors": {"model": [str(exc)]},
        }
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return JsonResponse(payload, status=500)
        return render(request, "recommender/form.html", {"form": submitted, "errors": payload["errors"]})

    disease = pred["predicted_disease"]

    meds = kb["medications"].get(disease, [])

    result = {
        "disease": disease,
        "confidence": pred["confidence"],
        "top_candidates": pred["top_candidates"],
        "description": kb["descriptions"].get(disease, "Description unavailable."),
        "medications": meds[:5],
        "precautions": kb["precautions"].get(disease, []),
        "diet": kb["diets"].get(disease, []),
        "workout": kb["workouts"].get(disease, []),
        "medical_disclaimer": "Decision support only. Consult a licensed physician before taking any medication.",
    }

    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return JsonResponse({"ok": True, "result": result})

    return render(request, "recommender/result.html", {"result": result, "form": _build_form()})
