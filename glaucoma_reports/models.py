from django.db import models

class OCTReport(models.Model):
    image = models.ImageField(upload_to='oct_reports/')
    min_gcl_ipl_od = models.FloatField(null=True, blank=True)
    min_gcl_ipl_os = models.FloatField(null=True, blank=True)
    avg_gcl_ipl_od = models.FloatField(null=True, blank=True)
    avg_gcl_ipl_os = models.FloatField(null=True, blank=True)
    rim_area_od = models.FloatField(null=True, blank=True)
    rim_area_os = models.FloatField(null=True, blank=True)
    vertical_cd_ratio_od = models.FloatField(null=True, blank=True)
    vertical_cd_ratio_os = models.FloatField(null=True, blank=True)
    
    # Prediction results
    glaucoma_prediction = models.CharField(max_length=20)
    prediction_confidence = models.FloatField()
    model_predictions = models.JSONField()
    
    # Explanation images
    heatmap_image = models.ImageField(upload_to='heatmaps/', null=True, blank=True)
    lime_image = models.ImageField(upload_to='lime/', null=True, blank=True)
    shap_image = models.ImageField(upload_to='shap/', null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"OCT Report {self.id} - {self.glaucoma_prediction} ({self.prediction_confidence:.2f})"