import streamlit as st
import tensorflow as tf
import os
import gdown
from PIL import Image
import numpy as np
from tensorflow.keras.layers import Rescaling

st.title("Classroom Classification AI Web App")

# Path to save/load the model
model_path = "models/my_custom_cnn.h5"

# Google Drive direct download link
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

# ✅ Match exactly the training class names
class_names = ['Chair', 'Keyboard', 'Monitor', 'Mouse', 'PC', 'Whiteboard ']

# Preprocess function
def preprocess_image(image, target_size=(224, 224)):
    img = image.convert('RGB').resize(target_size)
    x = np.array(img, dtype=np.float32)
    # Apply scaling only if the model does NOT have Rescaling
    has_rescaling = any(isinstance(l, Rescaling) for l in model.layers)
    if not has_rescaling:
        x = x / 255.0
    return np.expand_dims(x, axis=0)

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img_array = preprocess_image(image)

    # Run prediction
    prediction = model.predict(img_array)

    # Display raw probabilities
    st.write("Raw prediction probabilities:")
    for i, name in enumerate(class_names):
        st.write(f"{name}: {prediction[0][i]*100:.2f}%")

    # Predicted class
    pred_index = np.argmax(prediction[0])
    pred_name = class_names[pred_index]
    st.success(f"Predicted class: {pred_name}")
