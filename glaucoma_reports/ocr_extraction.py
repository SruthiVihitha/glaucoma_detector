from roboflow import Roboflow
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# Initialize once
rf = Roboflow(api_key="z8rBmkIxjqIa2aAm53sw")
project = rf.workspace().project("optical-character-recognition-3xdkm")
dataset = project.version("5").download("yolov8")

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def extract_relevant_metrics(image_path):
    # Load image
    img = Image.open(image_path)

    # Detect text boxes using YOLOv8 (you need to load your own trained model here)
    results = project.predict(image_path)
    extracted_text = {}

    for obj in results:
        cropped = img.crop(obj['bbox'])  # (left, upper, right, lower)
        pixel_values = processor(images=cropped, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Map text into metric names using regex or keys from obj['class']
        extracted_text[obj['class']] = text

    # Process values into float/int
    parsed_data = {
        'min_gcl_ipl_od': float(extracted_text.get('min_gcl_od', 65)),
        'min_gcl_ipl_os': float(extracted_text.get('min_gcl_os', 78)),
        'avg_gcl_ipl_od': float(extracted_text.get('avg_gcl_od', 70)),
        'avg_gcl_ipl_os': float(extracted_text.get('avg_gcl_os', 80)),
        'rim_area_od': float(extracted_text.get('rim_od', 1.0)),
        'rim_area_os': float(extracted_text.get('rim_os', 1.0)),
        'vertical_cd_ratio_od': float(extracted_text.get('cd_od', 0.4)),
        'vertical_cd_ratio_os': float(extracted_text.get('cd_os', 0.4)),
    }
    return parsed_data
