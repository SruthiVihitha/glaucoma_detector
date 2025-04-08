from django.contrib import admin
from .models import OCTReport

@admin.register(OCTReport)
class OCTReportAdmin(admin.ModelAdmin):
    list_display = ('id', 'glaucoma_prediction', 'prediction_confidence', 'created_at')
    list_filter = ('glaucoma_prediction', 'created_at')
    search_fields = ('glaucoma_prediction',)
    readonly_fields = ('created_at',)