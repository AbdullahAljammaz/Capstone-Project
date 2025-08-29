
import streamlit as st
import tensorflow as tf
import os
import gdown
from PIL import Image
import numpy as np

st.title("Classroom Classification AI Web App")

# Path to save/load the model
model_path = "models/my_custom_cnn.h5"

# Put your direct Google Drive link here
gdrive_url = "https://drive.google.com/uc?id=1OzRiSRs-k0L8B1dro4JyW5rjPv7Zzlnp"

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Download model if not exists
if not os.path.exists(model_path):
    st.info("Downloading model, please wait...")
    gdown.download(gdrive_url, model_path, quiet=False)
    st.success("Model downloaded!")

# Load the model
model = tf.keras.models.load_model(model_path)
st.success("Model loaded successfully!")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image (adjust size for your model)
    img_array = np.array(image.resize((224, 224)))
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Run prediction
    predictions = model.predict(img_array)
    st.write("Model predictions:", predictions)
