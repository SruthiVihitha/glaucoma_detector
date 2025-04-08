import tensorflow as tf
from tensorflow.keras.applications import ResNet50, InceptionV3, MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Multiply, Reshape
from tensorflow.keras.models import Model
import numpy as np
import cv2
import lime
from lime import lime_image
import shap
import matplotlib.pyplot as plt
from PIL import Image

# Squeeze-and-Excitation Block
def se_block(input_tensor, ratio=16):
    """SE-Net block implementation"""
    channels = input_tensor.shape[-1]
    se_shape = (1, 1, channels)
    
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = Reshape(se_shape)(se)
    se = Dense(channels // ratio, activation='relu')(se)
    se = Dense(channels, activation='sigmoid')(se)
    
    return Multiply()([input_tensor, se])

def build_models(input_shape=(224, 224, 3)):
    """Build all four model architectures"""
    models = {}
    
    # 1. ResNet50
    base_resnet = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_resnet.output
    x = GlobalAveragePooling2D()(x)
    predictions_resnet = Dense(1, activation='sigmoid')(x)
    models['resnet50'] = Model(inputs=base_resnet.input, outputs=predictions_resnet)
    
    # 2. SE-ResNet
    base_seresnet = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    for layer in base_seresnet.layers:
        if 'conv' in layer.name and 'bn' not in layer.name:
            x = layer.output
            x = se_block(x)
            layer._outbound_nodes = []
            layer.outbound_nodes = []
            x = layer(x)
    x = GlobalAveragePooling2D()(base_seresnet.output)
    predictions_seresnet = Dense(1, activation='sigmoid')(x)
    models['seresnet'] = Model(inputs=base_seresnet.input, outputs=predictions_seresnet)
    
    # 3. MobileNetV2
    base_mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_mobilenet.output
    x = GlobalAveragePooling2D()(x)
    predictions_mobilenet = Dense(1, activation='sigmoid')(x)
    models['mobilenetv2'] = Model(inputs=base_mobilenet.input, outputs=predictions_mobilenet)
    
    # 4. InceptionV3
    base_inception = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_inception.output
    x = GlobalAveragePooling2D()(x)
    predictions_inception = Dense(1, activation='sigmoid')(x)
    models['inceptionv3'] = Model(inputs=base_inception.input, outputs=predictions_inception)
    
    return models

def convert_models_to_tflite(models):
    """Convert all models to TFLite format"""
    tflite_models = {}
    for name, model in models.items():
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        with open(f'models/{name}_glaucoma.tflite', 'wb') as f:
            f.write(tflite_model)
        tflite_models[name] = f'models/{name}_glaucoma.tflite'
    return tflite_models

def load_tflite_models():
    """Load all TFLite models"""
    model_paths = {
        'resnet50': 'models/resnet50_glaucoma.tflite',
        'seresnet': 'models/seresnet_glaucoma.tflite',
        'mobilenetv2': 'models/mobilenetv2_glaucoma.tflite',
        'inceptionv3': 'models/inceptionv3_glaucoma.tflite'
    }
    
    models = {}
    for name, path in model_paths.items():
        interpreter = tf.lite.Interpreter(model_path=path)
        interpreter.allocate_tensors()
        models[name] = interpreter
    return models

def preprocess_image(image_path, target_size=(224, 224)):
    """Preprocess image for model input"""
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0
    return np.expand_dims(img, axis=0)

def predict_with_model(interpreter, image_array):
    """Make prediction using TFLite model"""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0][0]

def ensemble_predict(models, image_path):
    """Make predictions using all models"""
    img_array = preprocess_image(image_path)
    
    predictions = {}
    for name, interpreter in models.items():
        predictions[name] = float(predict_with_model(interpreter, img_array))
    
    # Weighted ensemble prediction
    weights = {
        'resnet50': 0.3,
        'seresnet': 0.3,
        'mobilenetv2': 0.2,
        'inceptionv3': 0.2
    }
    ensemble_pred = sum(predictions[name] * weights[name] for name in predictions)
    predictions['ensemble'] = ensemble_pred
    
    return predictions

def generate_gradcam_pp(interpreter, image_path, model_name):
    """Generate Grad-CAM++ heatmap"""
    img_array = preprocess_image(image_path)
    
    # Mock implementation - in practice would need original Keras model
    heatmap = np.random.rand(224, 224)  # Replace with actual Grad-CAM++ implementation
    
    original_img = cv2.imread(image_path)
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    return heatmap

def generate_lime_explanation(interpreter, image_path):
    """Generate LIME explanation"""
    def predict_fn(images):
        processed = []
        for img in images:
            img = (img * 255).astype(np.float32)
            img = tf.image.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0)
            processed.append(img)
        batch = np.vstack(processed)
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], batch)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])
    
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_array,
        predict_fn,
        top_labels=1,
        hide_color=0,
        num_samples=100
    )
    return explanation

def generate_shap_explanation(interpreter, image_path):
    """Generate SHAP explanation"""
    img_array = preprocess_image(image_path, target_size=(100, 100))
    
    def predict_fn(images):
        processed = []
        for img in images:
            img = tf.image.resize(img, (224, 224))
            img = np.expand_dims(img, axis=0)
            processed.append(img)
        batch = np.vstack(processed)
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], batch)
        interpreter.invoke()
        return interpreter.get_tensor(output_details[0]['index'])
    
    masker = shap.maskers.Image("inpaint_telea", img_array.shape[1:])
    explainer = shap.Explainer(predict_fn, masker)
    return explainer(img_array, max_evals=100)