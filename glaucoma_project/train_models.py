# train_models.py
import tensorflow as tf
from predictor import build_models, convert_models_to_tflite

# Load dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    'data/train',
    image_size=(224, 224),
    batch_size=32,
    label_mode='binary'
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    'data/val',
    image_size=(224, 224),
    batch_size=32,
    label_mode='binary'
)

# Build and train models
models = build_models()
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(train_ds, validation_data=val_ds, epochs=10)

# Convert to TFLite
convert_models_to_tflite(models)
print("Models converted to TFLite format!")