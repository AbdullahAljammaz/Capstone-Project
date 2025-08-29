import streamlit as st
import tensorflow as tf
import os
from PIL import Image
import numpy as np
import gdown

st.title("Classroom Classification AI Web App")

# Path to save/load the model
model_path = "models/my_custom_cnn.h5"

# Google Drive direct download link for the model
gdrive_model_url = "https://drive.google.com/uc?id=1OzRiSRs-k0L8B1dro4JyW5rjPv7Zzlnp"

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Download model if not exists
if not os.path.exists(model_path):
    st.info("Downloading model, please wait...")
    gdown.download(gdrive_model_url, model_path, quiet=False)
    st.success("Model downloaded!")

# Load the model
model = tf.keras.models.load_model(model_path)
st.success("Model loaded successfully!")

# Class names in correct order
class_names = ['Chair', 'Keyboard', 'Monitor', 'Mouse', 'PC', 'Whiteboard']

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Display original size
    st.write("Original image size (width, height):", image.size)
    
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Resize and normalize
    img_array = np.array(image.resize((224, 224)))
    st.write("Resized image shape:", img_array.shape)
    
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Run prediction
    prediction = model.predict(img_array)
    pred_index = np.argmax(prediction[0])
    pred_name = class_names[pred_index]

    # Display predicted class
    st.success(f"Predicted class: {pred_name}")
