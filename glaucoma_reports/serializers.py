from rest_framework import serializers
from .models import OCTReport

class OCTReportSerializer(serializers.ModelSerializer):
    class Meta:
        model = OCTReport
        fields = '__all__'
