import os
import json
from datetime import datetime
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from .models import OCTReport
from .predictor import (
    load_tflite_models,
    ensemble_predict,
    generate_gradcam_pp,
    generate_lime_explanation,
    generate_shap_explanation,
    preprocess_image
)
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import numpy as np

MODELS = load_tflite_models()

@csrf_exempt
def handle_image_upload(request):
    if request.method == 'POST' and request.FILES.get('oct_report'):
        try:
            uploaded_file = request.FILES['oct_report']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{uploaded_file.name}"
            file_path = os.path.join(settings.MEDIA_ROOT, 'oct_reports', filename)
            
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                for chunk in uploaded_file.chunks():
                    f.write(chunk)
            
            predictions = ensemble_predict(MODELS, file_path)
            final_prediction = 'Glaucoma Detected' if predictions['ensemble'] > 0.5 else 'No Glaucoma'
            
            oct_report = OCTReport(
                image=f'oct_reports/{filename}',
                glaucoma_prediction=final_prediction,
                prediction_confidence=predictions['ensemble'],
                model_predictions=predictions
            )
            
            generate_explanations(oct_report, file_path)
            oct_report.save()
            
            return JsonResponse({
                'status': 'success',
                'prediction': final_prediction,
                'confidence': predictions['ensemble'],
                'model_predictions': predictions,
                'explanations': {
                    'heatmap': f'/media/{oct_report.heatmap_image}' if oct_report.heatmap_image else None,
                    'lime': f'/media/{oct_report.lime_image}' if oct_report.lime_image else None,
                    'shap': f'/media/{oct_report.shap_image}' if oct_report.shap_image else None
                }
            })
            
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=500)
    
    return JsonResponse({'status': 'invalid_request'}, status=400)

def generate_explanations(oct_report, image_path):
    """Generate all explanation visualizations"""
    try:
        os.makedirs(os.path.join(settings.MEDIA_ROOT, 'heatmaps'), exist_ok=True)
        os.makedirs(os.path.join(settings.MEDIA_ROOT, 'lime'), exist_ok=True)
        os.makedirs(os.path.join(settings.MEDIA_ROOT, 'shap'), exist_ok=True)
        
        base_filename = os.path.basename(image_path)
        best_model_name = max(
            [(name, abs(pred - 0.5)) for name, pred in oct_report.model_predictions.items() 
             if name != 'ensemble'],
            key=lambda x: x[1]
        )[0]
        interpreter = MODELS[best_model_name]
        
        # Grad-CAM++
        heatmap = generate_gradcam_pp(interpreter, image_path, best_model_name)
        heatmap_path = f'heatmaps/gradcampp_{base_filename}'
        cv2.imwrite(os.path.join(settings.MEDIA_ROOT, heatmap_path), heatmap)
        oct_report.heatmap_image = heatmap_path
        
        # LIME Explanation
        lime_exp = generate_lime_explanation(interpreter, image_path)
        lime_path = f'lime/lime_{base_filename}'
        temp, mask = lime_exp.get_image_and_mask(0, positive_only=True, num_features=5)
        plt.imsave(os.path.join(settings.MEDIA_ROOT, lime_path), mark_boundaries(temp / 2 + 0.5, mask))
        oct_report.lime_image = lime_path
        
        # SHAP Explanation
        shap_exp = generate_shap_explanation(interpreter, image_path)
        shap_path = f'shap/shap_{base_filename}'
        shap.image_plot([shap_exp.values[0]], -shap_exp.data[0])
        plt.savefig(os.path.join(settings.MEDIA_ROOT, shap_path))
        plt.close()
        oct_report.shap_image = shap_path
        
    except Exception as e:
        print(f"Explanation generation failed: {str(e)}")