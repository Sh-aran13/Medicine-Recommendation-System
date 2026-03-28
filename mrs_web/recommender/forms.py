from django import forms


class RecommendationForm(forms.Form):
    age = forms.IntegerField(min_value=0, max_value=120, required=True)
    gender = forms.ChoiceField(
        choices=[("male", "Male"), ("female", "Female"), ("other", "Other")],
        required=True,
    )
    symptoms = forms.MultipleChoiceField(choices=[], required=True)
