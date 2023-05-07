from .utils import languages
from django import forms

class InputForm(forms.Form):
    input_field = forms.CharField(max_length=1000)

class SelectForm_to(forms.Form):
    select_field_to = forms.ChoiceField(choices=zip(languages, languages))

class SelectForm_from(forms.Form):
    select_field_from = forms.ChoiceField(choices=zip(languages, languages))
