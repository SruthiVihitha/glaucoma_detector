from django import forms
from .models import OCTReport

class OCTReportForm(forms.ModelForm):
    class Meta:
        model = OCTReport
        fields = ['image']
