import streamlit as st
from PIL import Image
import tempfile
import numpy as np
from io import BytesIO
import sys
import os

# Allow importing from parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import model functions
from glaucoma_reports.predictor import predict_glaucoma, generate_gradcam_pp_heatmap, load_models

# Load models once
resnet_model, _, _ = load_models()

# Sidebar navigation
st.sidebar.title("OCT Report Prediction")
sidebar_choice = st.sidebar.radio(
    "Select a section",
    ("Upload Image", "Prediction Results", "View Heatmap")
)

# Upload Section
if sidebar_choice == "Upload Image":
    st.header("Upload OCT Report Image")
    uploaded_image = st.file_uploader("Choose an OCT image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # Check the file type
        file_type = uploaded_image.type  # MIME type of the uploaded file
        file_name = uploaded_image.name  # File name with extension

        # Validate file type (must be jpeg or png)
        if file_type not in ["image/jpeg", "image/png"]:
            st.error("Please upload an image in JPEG or PNG format.")
        else:
            # Check if the media folder exists, create if not
            media_dir = os.path.join(os.getcwd(), 'media')
            if not os.path.exists(media_dir):
                os.makedirs(media_dir)

            # Save the image to the media folder
            image_path = os.path.join(media_dir, file_name)
            with open(image_path, 'wb') as f:
                f.write(uploaded_image.getbuffer())

            # Display the uploaded image
            st.image(uploaded_image, caption="Uploaded Image", use_container_width=True)

            # Store the path in session state to use in other sections
            st.session_state["uploaded_image_path"] = image_path
# Prediction Section
elif sidebar_choice == "Prediction Results":
    st.header("Prediction Results")
    if "uploaded_image" in st.session_state:
        uploaded_image = st.session_state["uploaded_image"]

        # Save to temp file for model input
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(uploaded_image.read())
            temp_image_path = tmp.name

        prediction = predict_glaucoma(temp_image_path, resnet_model)
        st.write("Raw Prediction Output:", prediction)  # Check the prediction
        prediction_value = prediction[0][0]
        threshold = 0.3  # Use a lower threshold for better sensitivity
        prediction_label = "Glaucoma Detected" if prediction_value > threshold else "No Glaucoma"
        confidence = round(float(prediction_value) * 100, 2)

        st.markdown(f"**Prediction:** {prediction_label}")
        st.markdown(f"**Confidence Score:** {confidence}%")

# Heatmap Section
# Heatmap Section
elif sidebar_choice == "View Heatmap":
    st.header("Grad-CAM++ Heatmap")
    if "uploaded_image_path" in st.session_state:
        image_path = st.session_state["uploaded_image_path"]

        try:
            # Open the image from the media directory using PIL
            img = Image.open(image_path)
            img = img.convert('RGB')  # Ensure it's in RGB mode
            img = img.resize((224, 224))  # Resize for model input

            # Debug: Display the image
            st.image(img, caption="Processed Image", use_container_width=True)

            # Generate the heatmap using the processed image
            heatmap = generate_gradcam_pp_heatmap(resnet_model, image_path)

            # Only display the heatmap if it was generated successfully
            if heatmap is not None:
                st.image(heatmap, caption="Grad-CAM++ Heatmap", use_container_width=True)
            else:
                st.error("Error: Unable to generate heatmap. Please check the model and image.")

        except IOError:
            st.error("Error: Unable to identify the uploaded image. Please upload a valid image.")
