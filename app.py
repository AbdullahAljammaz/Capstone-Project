import streamlit as st
import tensorflow as tf
import os
import gdown
from PIL import Image
import numpy as np
from tensorflow.keras.layers import Rescaling
import base64

# ===========================
# Background image function
# ===========================
def set_background(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Add background (replace with your image in repo)
set_background("Untitled design.png")  # <- image in the same folder as app.py

# ===========================
# App title
# ===========================
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

# ✅ Make sure this matches exactly train_ds.class_names
class_names = ['Chair', 'Keyboard', 'Monitor', 'Mouse', 'PC', 'Whiteboard ']

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # ✅ Preprocess image (no manual scaling here)
    img_array = np.array(image.resize((224, 224)), dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1,224,224,3)

    # Run prediction
    prediction = model.predict(img_array)

    # Display raw probabilities
    st.write("Raw prediction probabilities:")
    for i, name in enumerate(class_names):
        st.write(f"{name}: {prediction[0][i]*100:.2f}%")

    # Predicted class
    pred_index = np.argmax(prediction[0])
    pred_name = class_names[pred_index]
    st.write("Predicted class:", pred_name)
